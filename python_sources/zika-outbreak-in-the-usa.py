#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
#enter states
state = ['Alabama','Alaska','Arizona','Arkansas','California','Colorado','Connecticut','Delaware','District of Columbia','Florida','Georgia','Hawaii','Idaho','Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana','Maine','Maryland','Massachusetts','Michigan','Minnesota','Mississippi','Missouri','Montana','Nebraska','Nevada','New Hampshire','New Jersey','New Mexico','New York','North Carolina','North Dakota','Ohio','Oklahoma','Oregon','Pennsylvania','Rhode Island','South Carolina','South Dakota','Tennessee','Texas,Utah','Vermont','Virginia','Washington','West Virginia','Wisconsin','Wyoming','American Samoa','Puerto Rico','U.S. Virgin Islands']

#enter disease_rate
disease_rate = [3,1,3,0,51,6,0,0,3,107,2,3,0,7,3,1,2,2,1,1,12,10,7,6,2,2,0,0,11,0,61,8,0,3,1,5,3,2,0,1,47,4,3,13,12,2,4,0,1,4,73,519,45]
#use scatter plot for better analytical results
plt.scatter(state,disease_rate)
#label the entire model 
plt.xlabel('State')
plt.ylabel('disease in numbers')
plt.title('Anaysis of Zika Outbreak in the USA')

plt.show()