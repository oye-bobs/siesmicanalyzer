from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import pandas as pd

# Initialize USGS client
client = Client("USGS")

# Define time range and California region bounds
# Note: Using a fixed end_time for consistency, as future dates won't have events.
# For truly current data, you could use UTCDateTime() for end_time.
start_time = UTCDateTime("2025-03-26") # Start date for the last three months
end_time = UTCDateTime("2025-06-26") # Current date
minlatitude = 32.0
maxlatitude = 42.0
minlongitude = -125.0
maxlongitude = -114.0

# Fetch events
# Removed 'limit=500' to fetch all events in the specified range and magnitude.
# Be aware that this could be a very large number of events and take time.
events = client.get_events(starttime=start_time,
                            endtime=end_time,
                            minlatitude=minlatitude,
                            maxlatitude=maxlatitude,
                            minlongitude=minlongitude,
                            maxlongitude=maxlongitude,
                            minmagnitude=2.0, # Fetching microseismicity for nowcasting features
                            orderby="time-asc")

# Extract into DataFrame
records = []
for event in events:
    # Use .preferred_origin() and .preferred_magnitude() for consistency,
    # falling back to the first available if preferred isn't set.
    origin = event.preferred_origin() or event.origins[0]
    magnitude = event.preferred_magnitude() or event.magnitudes[0]
    
    # Ensure all required attributes exist before accessing
    time = origin.time.datetime if origin.time else None
    lat = origin.latitude if origin.latitude is not None else None
    lon = origin.longitude if origin.longitude is not None else None
    depth = origin.depth / 1000 if origin.depth is not None else None # Convert to km
    mag = magnitude.mag if magnitude.mag is not None else None
    mag_type = magnitude.magnitude_type if magnitude.magnitude_type else None

    records.append({
        "origin_time": time,
        "latitude": lat,
        "longitude": lon,
        "depth_km": depth,
        "magnitude": mag,
        "magnitude_type": mag_type
    })

df = pd.DataFrame(records)

# Output
print(f"Total events retrieved: {len(df)}")
print(df.head())

# Optional: Save to CSV for easier access later
df.to_csv("california_earthquakes_2005_2025_M2+.csv", index=False)