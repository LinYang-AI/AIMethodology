This is a MLflow practice for a naive ML model in production based on class <AI Methodology> of EPITA-2023 for AIS program.

## Data Dictionary

### Covariates

* `id`: building id
* `Year_Factor`: anonymized year in which the weather and energy usage factors were observed
* `State_Factor`: anonymized state in which the building is located
* `building_class`: building classification
* `facility_type`: building usage type
* `floor_area`: floor area (in square feet) of the building
* `year_built`: year in which the building was constructed
* `energy_star_rating`: the energy star rating of the building
* `ELEVATION`: elevation of the building location
* `january_min_temp`: minimum temperature in January (in Fahrenheit) at the location of the building
* `january_avg_temp`: average temperature in January (in Fahrenheit) at the location of the building
* `january_max_temp`: maximum temperature in January (in Fahrenheit) at the location of the building
* `cooling_degree_days`: cooling degree day for a given day is the number of degrees where the daily average temperature exceeds 65 degrees Fahrenheit. Each month is summed to produce an annual total at the location of the building.
* `heating_degree_days`: heating degree day for a given day is the number of degrees where the daily average temperature falls under 65 degrees Fahrenheit. Each month is summed to produce an annual total at the location of the building.
* `precipitation_inches`: annual precipitation in inches at the location of the building
* `snowfall_inches`: annual snowfall in inches at the location of the building
* `snowdepth_inches`: annual snow depth in inches at the location of the building
* `avg_temp`: average temperature over a year at the location of the building
* `days_below_30F`: total number of days below 30 degrees Fahrenheit at the location of the building
* `days_below_20F`: total number of days below 20 degrees Fahrenheit at the location of the building
* `days_below_10F`: total number of days below 10 degrees Fahrenheit at the location of the building
* `days_below_0F`: total number of days below 0 degrees Fahrenheit at the location of the building
* `days_above_80F`: total number of days above 80 degrees Fahrenheit at the location of the building
* `days_above_90F`: total number of days above 90 degrees Fahrenheit at the location of the building
* `days_above_100F`: total number of days above 100 degrees Fahrenheit at the location of the building
* `days_above_110F`: total number of days above 110 degrees Fahrenheit at the location of the building
* `direction_max_wind_speed`: wind direction for maximum wind speed at the location of the building. Given in 360-degree
  compass point directions (e.g. 360 = north, 180 = south, etc.).
* `direction_peak_wind_speed`: wind direction for peak wind gust speed at the location of the building. Given in 360-degree compass point directions (e.g. 360 = north, 180 = south, etc.).
* `max_wind_speed`: maximum wind speed at the location of the building
* `days_with_fog`: number of days with fog at the location of the building

### Target

* `site_eui`: Site Energy Usage Intensity is the amount of heat and electricity consumed by a building as reflected in utility bills
