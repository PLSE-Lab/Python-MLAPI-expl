import pandas as pd

books_file = '../input/books.csv'
ratings_file = '../input/ratings.csv'
to_read_file = '../input/to_read.csv'

books_to_show = 10

#

b = pd.read_csv( books_file )
r = pd.read_csv( ratings_file )
t = pd.read_csv( to_read_file )

while True:

	user_id = r.sample( 1 ).iloc[0].user_id
	print( "user", user_id )

	# books rated
	ur = r[ r.user_id == user_id ]
	if len( ur ) < 3:
		continue
	
	ur = ur.sort_values( 'rating', ascending = False )[:books_to_show]
	ur = ur.merge( b, on = 'book_id' )

	# books to read
	ut = t[ t.user_id == user_id ]
	if len( ut ) < 1:
		continue
	
	ut = ut[:books_to_show]
	ut = ut.merge( b, on = 'book_id' )
	break

print( 'highest rated:' )
print( ur[[ 'title', 'rating' ]] )

print( '\nto read:' )
print( ut[[ 'title' ]] )