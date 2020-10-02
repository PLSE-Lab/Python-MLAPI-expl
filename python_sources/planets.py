import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

print('Hello, world!')
planet_data = pd.read_csv('../input/oec.csv')
#print(data.columns)
#print(data.describe())

mass_data = planet_data['PlanetaryMassJpt'].dropna()
log_mass_data = np.log(mass_data)

log_mass_data.hist(alpha=0.8,bins=30).get_figure().savefig('Planet_masses.png')