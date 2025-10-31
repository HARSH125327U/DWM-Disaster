"""
Comprehensive Data Integration Pipeline for India Disaster Vulnerability Analysis
Implements the complete strategy from district-level integration to multi-hazard risk indices
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PHASE 1: DATA LOADING AND INITIAL SETUP
# ============================================================================

def load_all_datasets():
    """Load all required datasets"""
    print("📁 Loading datasets...")
    
    # Load district boundaries (base layer)
    districts = gpd.read_file('C:\\Users\\tanwa\\Downloads\\spatialDisaster\\spatialDisaster\\india_district.geojson')
    districts.rename(columns={
        'NAME_2': 'DISTRICT',
        'NAME_1': 'STATE',
        'NAME_0': 'COUNTRY'
    }, inplace=True)
    print("First few rows of DataFrame:\n", districts.head())
    print("DataFrame columns:", districts.columns.tolist())

    # Load population data
    population = pd.read_csv('C:\\Users\\tanwa\\Downloads\\spatialDisaster\\spatialDisaster\\district wise population and centroids.csv')
    population.rename(columns={
        'District': 'DISTRICT',
        'State': 'STATE'
    }, inplace=True)

    # Load earthquake datasets
    eq_2019_2021 = pd.read_csv('C:\\Users\\tanwa\\Downloads\\spatialDisaster\\spatialDisaster\\Indian_earthquake_data.csv')
    eq_2024_2025 = pd.read_csv('C:\\Users\\tanwa\\Downloads\\spatialDisaster\\spatialDisaster\\eq-latest_2025.csv')
    
    # Load historical disaster data
    historical = pd.read_csv('C:\\Users\\tanwa\\Downloads\\spatialDisaster\\spatialDisaster\\_India Earthquake-Tsunami Risk Assessment Dataset (2000-2025).csv')
    
    # Load flood risk data
    flood_risk = pd.read_csv('C:\\Users\\tanwa\\Downloads\\spatialDisaster\\spatialDisaster\\flood_risk_dataset_india.csv')

    print("✅ All datasets loaded successfully")
    return districts, population, eq_2019_2021, eq_2024_2025, historical, flood_risk

def standardize_district_names(df, district_col):
    """Standardize district names for consistent merging"""
    if district_col not in df.columns:
        print(f"⚠️ Warning: Column '{district_col}' not found in dataframe")
        print(f"Available columns: {df.columns.tolist()}")
        return df
    df[district_col] = df[district_col].str.strip().str.title()
    df[district_col] = df[district_col].str.replace(' And ', ' and ')
    df[district_col] = df[district_col].str.replace('  ', ' ')
    return df


# ============================================================================
# PHASE 2.1: POPULATION & DEMOGRAPHICS INTEGRATION
# ============================================================================

def integrate_population(districts, population):
    """Merge population data with district boundaries"""
    print("\n👥 Integrating population data...")

    pop_district_col = None
    for col in population.columns:
        if 'district' in col.lower():
            pop_district_col = col
            break
    
    if pop_district_col is None:
        print(f"⚠️ Warning: No district column found in population data")
        print(f"Available columns: {population.columns.tolist()}")
        return districts
    
    print(f"Found district column: {pop_district_col}")
   # Standardize names - only if column exists
    if 'DISTRICT' in districts.columns:
        districts = standardize_district_names(districts, 'DISTRICT')
    if pop_district_col in population.columns:
        population = standardize_district_names(population, pop_district_col)
    
    # Calculate area if not present
    if 'area_sq_km' not in districts.columns:
        districts['area_sq_km'] = districts.geometry.area / 1e6  # Convert to sq km
    
    # Find population and coordinate columns
    pop_col = next((col for col in population.columns if 'population' in col.lower()), None)
    lat_col = next((col for col in population.columns if 'lat' in col.lower()), None)
    lon_col = next((col for col in population.columns if 'lon' in col.lower()), None)
    
    print(f"Found columns - Population: {pop_col}, Latitude: {lat_col}, Longitude: {lon_col}")
    
    merge_cols = [pop_district_col]
    if pop_col:
        merge_cols.append(pop_col)
    if lat_col:
        merge_cols.append(lat_col)
    if lon_col:
        merge_cols.append(lon_col)
    
    # Merge population data
    districts = districts.merge(
        population[merge_cols], 
        left_on='DISTRICT',
        right_on=pop_district_col,
        how='left'
    )
    
    # Rename columns to standard names
    rename_dict = {}
    if pop_col:
        rename_dict[pop_col] = 'population_2011'
    if lat_col:
        rename_dict[lat_col] = 'centroid_lat'
    if lon_col:
        rename_dict[lon_col] = 'centroid_lon'
    
    districts.rename(columns=rename_dict, inplace=True)
    
    # Remove duplicate district column
    if pop_district_col in districts.columns and pop_district_col != 'DISTRICT':
        districts.drop(columns=[pop_district_col], inplace=True)
    
    # Calculate density
    if 'population_2011' in districts.columns:
        districts['population_density'] = districts['population_2011'] / districts['area_sq_km']
        print(f"✅ Population integrated for {districts['population_2011'].notna().sum()} districts")
    else:
        print("⚠️ Warning: Population column not found")
    
    return districts

# ============================================================================
# PHASE 2.2: EARTHQUAKE EVENT HISTORY INTEGRATION
# ============================================================================

def prepare_earthquake_data(eq_2019_2021, eq_2024_2025):
    """Combine and prepare earthquake datasets"""
    print("\n🌍 Preparing earthquake data...")
    
    # Standardize column names for 2019-2021 data
    eq_2019_2021_clean = eq_2019_2021.copy()
    # Map common column variations
    col_mapping_2019 = {}
    for col in eq_2019_2021.columns:
        col_lower = col.lower()
        if 'mag' in col_lower and 'mag' not in col_mapping_2019:
            col_mapping_2019['mag'] = col
        elif 'depth' in col_lower and 'depth' not in col_mapping_2019:
            col_mapping_2019['depth'] = col
        elif 'lat' in col_lower and 'latitude' not in col_mapping_2019:
            col_mapping_2019['latitude'] = col
        elif 'lon' in col_lower and 'longitude' not in col_mapping_2019:
            col_mapping_2019['longitude'] = col
    
    eq_2019_2021_clean.rename(columns=col_mapping_2019, inplace=True)

    # Add year column
    if 'time' in eq_2019_2021_clean.columns:
        try:
            eq_2019_2021_clean['year'] = pd.to_datetime(eq_2019_2021_clean['time'], errors='coerce').dt.year
        except:
            eq_2019_2021_clean['year'] = 2020  # Default year
    else:
        eq_2019_2021_clean['year'] = 2020
    # Standardize 2024-2025 data
    eq_2024_2025_clean = eq_2024_2025.copy()
    col_mapping_2024 = {}
    for col in eq_2024_2025.columns:
        col_lower = col.lower()
        if 'mag' in col_lower and 'mag' not in col_mapping_2024:
            col_mapping_2024['mag'] = col
        elif 'depth' in col_lower and 'depth' not in col_mapping_2024:
            col_mapping_2024['depth'] = col
        elif 'lat' in col_lower and 'latitude' not in col_mapping_2024:
            col_mapping_2024['latitude'] = col
        elif 'lon' in col_lower and 'longitude' not in col_mapping_2024:
            col_mapping_2024['longitude'] = col
        elif 'time' in col_lower or 'date' in col_lower:
            col_mapping_2024['time'] = col
    
    eq_2024_2025_clean.rename(columns=col_mapping_2024, inplace=True)
    if 'time' in eq_2024_2025_clean.columns:
        try:
            eq_2024_2025_clean['year'] = pd.to_datetime(eq_2024_2025_clean['time'], errors='coerce').dt.year
        except:
            eq_2024_2025_clean['year'] = 2024  # Default year
    else:
        eq_2024_2025_clean['year'] = 2024

    required_cols = ['mag', 'depth', 'latitude', 'longitude', 'year']
    for col in required_cols:
        if col not in eq_2019_2021_clean.columns:
            eq_2019_2021_clean[col] = np.nan
        if col not in eq_2024_2025_clean.columns:
            eq_2024_2025_clean[col] = np.nan
            
    eq_2019_2021_final = eq_2019_2021_clean[required_cols].copy()
    eq_2024_2025_final = eq_2024_2025_clean[required_cols].copy()
    # Combine datasets
    eq_combined = pd.concat([eq_2019_2021_final, eq_2024_2025_final], ignore_index=True)

    # Create geometry
    eq_combined = eq_combined.dropna(subset=['latitude', 'longitude'])
    if len(eq_combined) == 0:
        print("⚠️ Warning: No valid earthquake coordinates found")
        return gpd.GeoDataFrame()
    geometry = [Point(xy) for xy in zip(eq_combined.longitude, eq_combined.latitude)]
    eq_gdf = gpd.GeoDataFrame(eq_combined, geometry=geometry, crs='EPSG:4326')
    
    print(f"✅ Combined {len(eq_gdf)} earthquake events")
    return eq_gdf


def spatial_join_earthquakes(districts, eq_gdf):
    """Perform spatial join and aggregate earthquake statistics by district"""
    print("\n📍 Performing spatial join...")
    
    # Ensure same CRS
    districts = districts.to_crs('EPSG:4326')
    eq_gdf = eq_gdf.to_crs('EPSG:4326')
    
    # Spatial join
    eq_districts = gpd.sjoin(eq_gdf, districts[['DISTRICT', 'geometry']], 
                             how='left', predicate='within')
    
    print(f"✅ Assigned {eq_districts['DISTRICT'].notna().sum()} earthquakes to districts")
    return eq_districts


def aggregate_earthquake_stats(districts, eq_districts):
    """Calculate comprehensive earthquake statistics for each district"""
    print("\n📊 Aggregating earthquake statistics...")
    if len(eq_districts) == 0 or 'DISTRICT' not in eq_districts.columns:
        print("⚠️ No earthquake data to aggregate")
        eq_cols = ['eq_count_total', 'eq_magnitude_max', 'eq_magnitude_mean', 
                   'eq_depth_mean', 'eq_frequency_per_year']
        for col in eq_cols:
            districts[col] = 0
        return districts
    agg_dict = {}
    if 'mag' in eq_districts.columns:
        agg_dict['mag'] = ['count', 'max', 'mean']
    if 'depth' in eq_districts.columns:
        agg_dict['depth'] = 'mean'
    # Overall statistics
    eq_stats = eq_districts.groupby('DISTRICT').agg(agg_dict).reset_index()

    new_cols = ['DISTRICT']
    for col in eq_stats.columns[1:]:
        if isinstance(col, tuple):
            if col[1] == 'count':
                new_cols.append('eq_count_total')
            elif col[1] == 'max':
                new_cols.append('eq_magnitude_max')
            elif col[1] == 'mean' and col[0] == 'mag':
                new_cols.append('eq_magnitude_mean')
            elif col[1] == 'mean' and col[0] == 'depth':
                new_cols.append('eq_depth_mean')
            else:
                new_cols.append('_'.join(map(str, col)))
        else:
            new_cols.append(col)
    eq_stats.columns = new_cols

    if 'time' in eq_districts.columns:
        recent_dates = eq_districts.groupby('DISTRICT')['time'].max()
        eq_stats['eq_recent_date'] = eq_stats['DISTRICT'].map(recent_dates)
    # Yearly counts
    if 'year' in eq_districts.columns:
        yearly = eq_districts.groupby(['DISTRICT', 'year']).size().unstack(fill_value=0)
        for year in yearly.columns:
            eq_stats[f'eq_count_{year}'] = eq_stats['DISTRICT'].map(yearly[year])
    
     # Calculate frequency per year
    if 'year' in eq_districts.columns:
        year_range = eq_districts['year'].max() - eq_districts['year'].min() + 1
        if 'eq_count_total' in eq_stats.columns:
            eq_stats['eq_frequency_per_year'] = eq_stats['eq_count_total'] / max(year_range, 1)
        else:
            eq_stats['eq_frequency_per_year'] = 0
    else:
        eq_stats['eq_frequency_per_year'] = 0
    
   # Magnitude classes
    if 'mag' in eq_districts.columns:
        eq_districts['mag_class'] = pd.cut(eq_districts['mag'], 
                                           bins=[0, 4.0, 5.0, 6.0, 10.0],
                                           labels=['minor', 'moderate', 'strong', 'major'])
        
        mag_classes = eq_districts.groupby(['DISTRICT', 'mag_class']).size().unstack(fill_value=0)
        for col in ['minor', 'moderate', 'strong', 'major']:
            if col in mag_classes.columns:
                eq_stats[f'eq_count_{col}'] = eq_stats['DISTRICT'].map(mag_classes[col])
            else:
                eq_stats[f'eq_count_{col}'] = 0
    
    # Merge with districts
    districts = districts.merge(eq_stats, on='DISTRICT', how='left')
    
    # Fill NaN with 0 for districts with no earthquakes
    eq_cols = [col for col in districts.columns if col.startswith('eq_')]
    districts[eq_cols] = districts[eq_cols].fillna(0)
    
    print(f"✅ Earthquake statistics calculated for all districts")
    return districts


def calculate_exposure_zones(districts, eq_districts):
    """Calculate population exposure within earthquake buffer zones"""
    print("\n🎯 Calculating exposure zones...")
    
    # Filter significant earthquakes (M >= 5.0)
    significant_eq = eq_districts[eq_districts['mag'] >= 5.0].copy()
    
    if len(significant_eq) == 0:
        districts['eq_exposure_50km'] = 0
        districts['eq_exposure_100km'] = 0
        return districts
    
    # Create buffers (50km and 100km)
    significant_eq_proj = significant_eq.to_crs('EPSG:7755')  # India-specific projection
    districts_proj = districts.to_crs('EPSG:7755')
    
    buffer_50km = significant_eq_proj.geometry.buffer(50000).unary_union
    buffer_100km = significant_eq_proj.geometry.buffer(100000).unary_union
    
    # Calculate exposure
    districts['eq_exposure_50km'] = districts_proj.geometry.intersection(buffer_50km).area / \
                                     districts_proj.geometry.area * districts['population_2011']
    districts['eq_exposure_100km'] = districts_proj.geometry.intersection(buffer_100km).area / \
                                      districts_proj.geometry.area * districts['population_2011']
    
    print("✅ Exposure zones calculated")
    return districts


# ============================================================================
# PHASE 2.3: HISTORICAL DISASTER VULNERABILITY
# ============================================================================

def integrate_historical_disasters(districts, historical):
    """Integrate historical disaster data"""
    print("\n📜 Integrating historical disaster data...")
    region_col = None
    for col in historical.columns:
        if 'region' in col.lower() or 'state' in col.lower():
            region_col = col
            break
    
    if region_col is None:
        print("⚠️ Warning: No region column found in historical data")
        districts['historical_damage_index'] = 0
        return districts
    
    mag_col = next((col for col in historical.columns if 'mag' in col.lower()), None)
    casualty_col = next((col for col in historical.columns if 'casual' in col.lower()), None)
    damage_col = next((col for col in historical.columns if 'damage' in col.lower() or 'economic' in col.lower()), None)
    tsunami_col = next((col for col in historical.columns if 'tsunami' in col.lower()), None)
    
    print(f"Found columns - Region: {region_col}, Magnitude: {mag_col}, Casualties: {casualty_col}, Damage: {damage_col}, Tsunami: {tsunami_col}")

    # Build aggregation dict based on available columns
    agg_dict = {}
    if mag_col:
        agg_dict[mag_col] = ['count', 'max', 'mean']
    if casualty_col:
        agg_dict[casualty_col] = 'sum'
    if damage_col:
        agg_dict[damage_col] = 'sum'
    if tsunami_col:
        agg_dict[tsunami_col] = lambda x: (x == 'Yes').any() if x.dtype == 'object' else x.any()
    
    if len(agg_dict) == 0:
        print("⚠️ Warning: No relevant historical data columns found")
        districts['historical_damage_index'] = 0
        return districts
    
   # Aggregate by state/region
    if 'STATE' in districts.columns:
        state_stats = historical.groupby(region_col).agg(agg_dict).reset_index()
    
    # Flatten column names
        new_cols = [region_col]
        for col in state_stats.columns[1:]:
            if isinstance(col, tuple):
                if col[1] == 'count':
                    new_cols.append('historical_events_state')
                elif col[1] == 'max':
                    new_cols.append('max_magnitude_state')
                elif col[1] == 'mean':
                    new_cols.append('mean_magnitude_state')
                elif col[1] == 'sum' and 'casual' in col[0].lower():
                    new_cols.append('total_casualties_state')
                elif col[1] == 'sum' and ('damage' in col[0].lower() or 'economic' in col[0].lower()):
                    new_cols.append('total_economic_damage_state')
                elif col[1] == '<lambda>':
                    new_cols.append('tsunami_risk')
                else:
                    new_cols.append('_'.join(map(str, col)))
            else:
                new_cols.append(col)
        
        state_stats.columns = new_cols

        # Rename region column to STATE for merging
        state_stats.rename(columns={region_col: 'STATE'}, inplace=True)
        
        # Merge with districts
        districts = districts.merge(state_stats, on='STATE', how='left')
        
        # Calculate historical damage index
        damage_cols = ['total_casualties_state', 'total_economic_damage_state', 
                       'historical_events_state', 'max_magnitude_state']
        available_damage_cols = [col for col in damage_cols if col in districts.columns]
        
        if len(available_damage_cols) > 0:
            # Ensure columns are numeric and fill NaN
            for col in available_damage_cols:
                districts[col] = pd.to_numeric(districts[col], errors='coerce').fillna(0)
            
            # Check if we have any non-zero values to normalize
            damage_matrix = districts[available_damage_cols].values
            if damage_matrix.max() > 0:
                scaler = MinMaxScaler()
                normalized = scaler.fit_transform(damage_matrix)
                districts['historical_damage_index'] = normalized.mean(axis=1) * 100
            else:
                districts['historical_damage_index'] = 0
            
            print(f"✅ Historical disaster data integrated ({len(available_damage_cols)} columns)")
        else:
            districts['historical_damage_index'] = 0
            print("⚠️ Warning: Could not calculate historical damage index")
    else:
        districts['historical_damage_index'] = 0
    
    return districts


# ============================================================================
# PHASE 2.4: FLOOD RISK INTEGRATION
# ============================================================================

def integrate_flood_risk(districts, flood_risk):
    """Integrate flood risk data"""
    print("\n🌊 Integrating flood risk data...")

    # Find district column in flood risk data
    flood_district_col = None
    for col in flood_risk.columns:
        if 'district' in col.lower():
            flood_district_col = col
            break
    
    if flood_district_col is None:
        print("⚠️ Warning: No district column found in flood risk data")
        districts['coastal_district'] = False
        return districts
    
    # Standardize district names
    flood_risk = standardize_district_names(flood_risk, 'District')
    
    # Select relevant columns
    flood_cols = [flood_district_col]
    for col in flood_risk.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['flood', 'risk', 'severity', 'monsoon', 'event']):
            flood_cols.append(col)
    
    if len(flood_cols) > 1:
        districts = districts.merge(flood_risk[flood_cols], 
                                   left_on='DISTRICT', right_on=flood_district_col, 
                                   how='left')
        if flood_district_col in districts.columns and flood_district_col != 'DISTRICT':
            districts.drop(columns=[flood_district_col], inplace=True)
        
        print(f"✅ Flood risk data integrated ({len(flood_cols)-1} columns)")
    else:
        print("⚠️ Warning: No relevant flood risk columns found")
    # Add coastal proximity flag
    # Simplified: assume districts with coastal keywords are coastal
    coastal_keywords = ['coastal', 'port', 'beach', 'island', 'sea', 'bay', 'ocean']
    districts['coastal_district'] = districts['DISTRICT'].str.lower().str.contains(
        '|'.join(coastal_keywords), na=False
    )

    return districts

# ============================================================================
# PHASE 5: MAIN INTEGRATION PIPELINE
# ============================================================================

def run_integration_pipeline():
    """Execute the complete integration pipeline"""
    print("=" * 80)
    print("🚀 STARTING INDIA DISASTER VULNERABILITY INTEGRATION PIPELINE")
    print("=" * 80)
    
    # Phase 1: Load data
    districts, population, eq_2019_2021, eq_2024_2025, historical, flood_risk = load_all_datasets()
    
    # Phase 2.1: Population integration
    districts = integrate_population(districts, population)
    
    # Phase 2.2: Earthquake integration
    eq_gdf = prepare_earthquake_data(eq_2019_2021, eq_2024_2025)
    eq_districts = spatial_join_earthquakes(districts, eq_gdf)
    districts = aggregate_earthquake_stats(districts, eq_districts)
    districts = calculate_exposure_zones(districts, eq_districts)
    
    # Phase 2.3: Historical disasters
    districts = integrate_historical_disasters(districts, historical)
    
    # Phase 2.4: Flood risk
    districts = integrate_flood_risk(districts, flood_risk)
    
    # Full integrated dataset
    districts.to_file('india_disaster_risk_integrated.geojson', driver='GeoJSON')
    print("✅ Saved: india_disaster_risk_integrated.geojson")
    
    # Simplified version (key columns only)
    key_columns = ['DISTRICT', 'STATE', 'geometry', 'population_2011', 'population_density',
                   'eq_count_total', 'eq_magnitude_max', 'eq_hazard_score', 
                   'exposure_score', 'MHVI', 'flood_risk_index', 'coastal_district']
    available_key_cols = [col for col in key_columns if col in districts.columns]
    districts[available_key_cols].to_file('india_disaster_risk_simplified.geojson', driver='GeoJSON')
    print("✅ Saved: india_disaster_risk_simplified.geojson")
    
    # Vulnerability rankings CSV
    ranking_cols = ['DISTRICT', 'STATE', 'MHVI', 'eq_hazard_score', 'exposure_score', 
                    'population_2011', 'eq_count_total']
    available_ranking_cols = [col for col in ranking_cols if col in districts.columns]
    rankings = districts[available_ranking_cols].sort_values('MHVI', ascending=False)
    rankings.to_csv('district_vulnerability_rankings.csv', index=False)
    print("✅ Saved: district_vulnerability_rankings.csv")
    
    # Full earthquake events with district assignment
    eq_districts_export = eq_districts[['DISTRICT', 'latitude', 'longitude', 'mag', 
                                        'depth', 'time', 'year']].copy()
    eq_districts_export.to_csv('earthquake_events_full.csv', index=False)
    print("✅ Saved: earthquake_events_full.csv")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("📊 INTEGRATION SUMMARY")
    print("=" * 80)
    print(f"Total districts: {len(districts)}")
    print(f"Districts with population data: {districts['population_2011'].notna().sum()}")
    print(f"Districts with earthquakes: {(districts['eq_count_total'] > 0).sum()}")
    print(f"Total earthquakes processed: {len(eq_districts)}")
    print(f"Average MHVI: {districts['MHVI'].mean():.2f}")
    print(f"Highest risk district: {districts.loc[districts['MHVI'].idxmax(), 'DISTRICT']}")
    print(f"MHVI range: {districts['MHVI'].min():.2f} - {districts['MHVI'].max():.2f}")
    
    print("\n✅ INTEGRATION PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    return districts, eq_districts


# ============================================================================
# EXECUTE PIPELINE
# ============================================================================

if __name__ == "__main__":
    integrated_districts, earthquake_events = run_integration_pipeline()
    
    # Display top 10 most vulnerable districts
    print("\n🔝 TOP 10 MOST VULNERABLE DISTRICTS:")
    print(integrated_districts.nlargest(10, 'MHVI')[['DISTRICT', 'STATE', 'MHVI', 
                                                       'eq_hazard_score', 'exposure_score']])