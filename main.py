from recommender import MovieRecommender

if __name__ == "__main__":
    user_input = [
        {'title': 'Conjuring, The', 'rating': 5},
        {'title': 'Avengers, The', 'rating': 4.1},
        {'title': 'Avengers: Age of Ultron', 'rating': 4.2},
        {'title': "Inception", 'rating': 4.8},
        {'title': 'Harry Potter and the Order of the Phoenix', 'rating': 4.5}
    ]

    mov = MovieRecommender()
    print(mov.predict(user_input))
