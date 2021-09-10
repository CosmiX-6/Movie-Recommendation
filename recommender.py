import pandas as pd


class MovieRecommender:

    def __init__(self):
        print('Initializing recommendation system...')
        # read the movies data
        movie_df = pd.read_csv('assets/movies.csv', usecols=['movieId', 'title', 'genres'])
        # read the movies rating data
        rating_df = pd.read_csv('assets/ratings.csv', usecols=['userId', 'movieId', 'rating', 'timestamp'])
        # extracting year from movies title
        movie_df['year'] = movie_df['title'].str.extract('(\(\d\d\d\d\))', expand=False).apply(lambda x: str(x)[-5:-1])
        # cleaning movie title
        movie_df['title'] = movie_df['title'].str.replace('(\(\d\d\d\d\))', '', regex=True)
        movie_df['title'] = movie_df['title'].apply(lambda x: x.strip())
        movie_df['genres'] = movie_df['genres'].str.split('|')
        movies_ = movie_df.copy()
        # creating separate genre column for each movie i.e creating dummies
        for index, row in movie_df.iterrows():
            for genre in row['genres']:
                movies_.at[index, genre] = 1
        # filling 'Nan' with 0
        self.movie_df = movie_df
        self.movies_ = movies_.fillna(0)
        self.rating_df = rating_df.drop('timestamp', 1)
        # creating genre table
        # setting movie_id as index
        genre_table = self.movies_.set_index(self.movies_['movieId'])
        # deleting column excluding genre
        self.genre_table = genre_table.drop(['title', 'movieId', 'genres', 'year'], axis=1)
        print('Ready for serving.')

    def calculate_interest(self, *args):
        input_movies = pd.DataFrame(*args)
        # searching movie in movies list
        get_id = self.movies_[self.movies_['title'].isin(input_movies['title'].tolist())]
        # merging the movie data into user input data
        input_movies = pd.merge(input_movies, get_id)
        user_movie = input_movies.drop(['title', 'rating', 'movieId', 'genres', 'year'], axis=1)
        interest_list = user_movie.transpose().dot(input_movies['rating'])

        return interest_list

    def predict(self, *args):
        # lets get movie for recommendation
        interest_list = self.calculate_interest(*args)
        recommendation_df = ((self.genre_table * interest_list).sum(axis=1)) / (interest_list.sum())
        # sorting our recommendations in descending order
        recommendation_df = recommendation_df.sort_values(ascending=False)
        return self.movie_df[self.movie_df['movieId'].isin(recommendation_df.head(20).index)]
