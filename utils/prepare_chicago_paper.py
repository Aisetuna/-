"""Reconstruct Chicago counts from public records; not an author-released dataset."""
import argparse
import json
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

START = pd.Timestamp('2022-09-01')
END = pd.Timestamp('2022-12-01')
SHAPE = (8736, 77)


def aggregate(frame, time_column, areas, counts, stats, date_format=None):
    times = pd.to_datetime(frame[time_column], errors='coerce', format=date_format)
    monthly = stats.setdefault('source_months', {})
    for month, count in times.dt.strftime('%Y-%m').value_counts().items():
        monthly[month] = monthly.get(month, 0) + int(count)
    areas = pd.to_numeric(areas, errors='coerce')
    in_period = times.ge(START) & times.lt(END)
    valid = in_period & areas.between(1, 77) & areas.eq(areas.round())
    stats['rows'] += len(frame)
    stats['in_period'] += int(in_period.sum())
    stats['accepted'] += int(valid.sum())
    steps = ((times[valid] - START) // pd.Timedelta(minutes=15)).to_numpy(dtype=int)
    np.add.at(counts, (steps, areas[valid].to_numpy(dtype=int) - 1), 1)


def process(raw, channel, regions, source_root, taxi_archive=None):
    counts = np.zeros(SHAPE, dtype=np.int32)
    stats = dict(rows=0, in_period=0, accepted=0)
    if channel == 'taxi' and taxi_archive is not None and taxi_archive.suffix.lower() == '.csv':
        stats['excluded_reversed_time_valid_area'] = 0
        for frame in pd.read_csv(taxi_archive, usecols=['Trip Start Timestamp', 'Trip End Timestamp', 'Dropoff Community Area'], chunksize=200000):
            fmt = '%m/%d/%Y %I:%M:%S %p'
            start = pd.to_datetime(frame['Trip Start Timestamp'], format=fmt, errors='coerce')
            end = pd.to_datetime(frame['Trip End Timestamp'], format=fmt, errors='coerce')
            area = pd.to_numeric(frame['Dropoff Community Area'], errors='coerce')
            reversed_time = start.gt(end)
            stats['excluded_reversed_time_valid_area'] += int((reversed_time & end.ge(START) & end.lt(END) & area.between(1, 77) & area.eq(area.round())).sum())
            aggregate(frame, 'Trip End Timestamp', area.mask(reversed_time), counts, stats, fmt)
            if stats['rows'] % 1000000 == 0:
                print('taxi', stats['rows'], 'accepted', stats['accepted'], flush=True)
    elif channel == 'crime':
        for frame in pd.read_csv(source_root / 'crime/2022/crimes-2022.csv', chunksize=200000):
            aggregate(frame, 'Date', frame['Community Area'], counts, stats,
                      '%m/%d/%Y %I:%M:%S %p')
    else:
        if channel == 'taxi' and taxi_archive is None:
            raise ValueError('Provide --taxi-archive pointing to a complete source ZIP covering September–November 2022')
        archives = sorted((source_root / 'bike/2022_09_11').glob('*.zip')) if channel == 'bike' else [taxi_archive]
        if channel == 'bike' and len(archives) != 3:
            raise ValueError('Expected all three Divvy monthly archives')
        for path in archives:
            with zipfile.ZipFile(path) as archive:
                members = [n for n in archive.namelist() if n.endswith('.csv') and not n.startswith('__MACOSX/')]
                if not members:
                    raise ValueError(f'No CSV in {path}')
                for member in members:
                    with archive.open(member) as source:
                        usecols = ['Trip End Timestamp', 'Dropoff Community Area'] if channel == 'taxi' else None
                        for frame in pd.read_csv(source, chunksize=200000, low_memory=False, usecols=usecols):
                            if channel == 'bike':
                                times = pd.to_datetime(frame['ended_at'], errors='coerce')
                                eligible = times.ge(START) & times.lt(END) & frame.end_lng.notna() & frame.end_lat.notna()
                                points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(frame.loc[eligible, 'end_lng'], frame.loc[eligible, 'end_lat']), index=frame.index[eligible], crs='EPSG:4326')
                                joined = gpd.sjoin(points, regions[['area', 'geometry']], how='left', predicate='within')
                                if joined.index.has_duplicates:
                                    raise ValueError('Ambiguous region assignment')
                                areas = joined['area'].reindex(frame.index)
                                aggregate(frame, 'ended_at', areas, counts, stats)
                            else:
                                names = {c.lower().replace(' ', '_'): c for c in frame.columns}
                                time_col = names['trip_end_timestamp']
                                area_col = names['dropoff_community_area']
                                aggregate(frame, time_col, frame[area_col], counts, stats, '%m/%d/%Y %I:%M:%S %p')
                                if stats['rows'] % 1000000 == 0:
                                    print(channel, stats, flush=True)
                    print(channel, member, stats, flush=True)
    stats['in_period_excluded_area'] = stats['in_period'] - stats['accepted'] - stats.get('excluded_reversed_time_valid_area', 0)
    stats['nonzero_bins'] = int(np.count_nonzero(counts))
    stats['daily_totals'] = counts.reshape(91, 96, 77).sum(axis=(1, 2)).tolist()
    if not counts.any():
        raise ValueError(f'{channel}: no usable records in requested period')
    (raw / f'{channel}_quality.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    if not all(stats['daily_totals']):
        raise ValueError(f'{channel}: missing entire days in requested period; inspect quality JSON before use')
    np.save(raw / f'{channel}_counts.npy', counts)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channels', nargs='+', choices=['taxi', 'bike', 'crime'], default=['taxi', 'bike', 'crime'])
    parser.add_argument('--taxi-archive', type=Path, help='Complete taxi CSV or ZIP source')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    raw = root / 'datasets/raw/chicago_paper_2022'
    raw.mkdir(parents=True, exist_ok=True)
    source_root = root / 'datasets/source_files'
    geo = source_root / 'geography/Chicago_Community_Areas.geojson'
    regions = gpd.read_file(geo)
    regions['area'] = regions.area_numbe.astype(int)
    regions = regions.sort_values('area').reset_index(drop=True).to_crs('EPSG:4326')
    assert regions['area'].tolist() == list(range(1, 78))
    for channel in args.channels:
        process(raw, channel, regions, source_root, args.taxi_archive)
    if not all((raw / f'{c}_counts.npy').exists() for c in ['taxi', 'bike', 'crime']):
        print('Partial channels saved; full dataset awaits remaining sources.')
        return
    data = np.stack([np.load(raw / f'{c}_counts.npy') for c in ['taxi', 'bike', 'crime']], axis=-1)
    assert data.shape == (8736, 77, 3) and np.isfinite(data).all() and (data >= 0).all()
    np.save(raw / 'chicago_taxi_bike_crime_2022_09_11_TND.npy', data)
    # Same Gaussian distance kernel as utils/get_adj_mat.py, using projected centroids.
    centers = regions.to_crs('EPSG:26916').centroid
    xy = np.column_stack([centers.x, centers.y])
    distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    adjacency = np.exp(-np.square(distance / distance.std())).astype(np.float32)
    adjacency[adjacency < 0.1] = 0
    out = root / 'datasets/chicago_15min'
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'chicago.npy', adjacency)
    metadata = dict(shape=list(data.shape), channels=['taxi_dropoffs', 'bike_returns', 'crime_incidents'], start=str(START), end_exclusive=str(END), interval_minutes=15, community_areas=list(range(1, 78)), time_basis='Local wall-clock timestamps; repeated DST hour combined', missing_policy='Invalid/missing areas discarded; empty bins zero-filled', adjacency='EPSG:26916 centroid Gaussian kernel exp(-(distance/std)^2), threshold 0.1', limitation='Public-source reconstruction, not verified identical to paper data; historical crime mirror may omit later reports; source timestamps and filtering may differ.', sources=dict(taxi='https://www.kaggle.com/datasets/javiertorresvergara/taxi-trips-chicago', bike='https://divvy-tripdata.s3.amazonaws.com/', crime='https://raw.githubusercontent.com/RandomFractals/chicago-crimes/main/data/crimes-2022.csv'), quality={c: json.loads((raw / f'{c}_quality.json').read_text()) for c in ['taxi', 'bike', 'crime']})
    if args.taxi_archive:
        metadata['taxi_archive'] = str(args.taxi_archive.resolve())
        metadata['sources']['taxi'] = 'User-provided archive; original download URL not independently verified'
    (raw / 'provenance.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Saved complete dataset', data.shape, 'totals', data.sum(axis=(0, 1)))


if __name__ == '__main__':
    main()
