# -*- coding: utf-8 -*-

import time
import random
import argparse
import pandas as pd
from collfilter import CollaborativeFilter


def load_files(ratings, items):
    """ This is just a example load function, made for the MovieLens
    collection, you can write your own for your own collection 
    """

    user_names = ("user_id", "item_id", "rating", "timestamp")
    items_names = ("item_id","movie_title","release_date","video_release_date",
        "IMDb_URL","unknown","Action","Adventure","Animation",
        "Children's","Comedy","Crime","Documentary","Drama","Fantasy",
        "Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi",
        "Thriller","War","Western")

    user = pd.read_csv(ratings, sep='\t', names=user_names, encoding="ISO-8859-1")
    items = pd.read_csv(items, sep='|', names=items_names, encoding="ISO-8859-1")

    # Get just important columns and drop the others:
    user = user[["user_id", "item_id", "rating"]]
    items = items[["item_id", "movie_title"]]

    return user, items


def get_user_ratings(films, user_id, n_films, rand=False):
    """ Ask active user to rate, from 1 to 5, `n_films` random films from
    `films` DataFrame. Returns a new dataframe with the same format as
        user_ratings
    user_id -- the id that will be assigned to the active user. It must
        be different from the others present in your data
    rand -- If True, active user rates will be generated randomly 
    """
    random.seed(89)
    n = films.shape[0]
    df = pd.DataFrame(columns=("user_id", "item_id", "rating"))
    for _ in range(n_films):
        index = random.randint(0, n-1)
        film_title = films.iloc[index]["movie_title"]
        film_id = films.iloc[index]["item_id"]

        if not rand:
            rate = input('What do you think about "%s"? [1-5]: ' % film_title)
            while type(rate) is str:
                try:
                    rate = int(rate)
                    if not (1 <= rate and rate <= 5): raise Exception
                except Exception:
                    rate = input("Oops! Something went wrong, ensure that your" + 
                        "rate is between 1 and 5: ")
        else:
            rate = random.randint(1,5)

        df = df.append({'user_id': user_id, 'item_id': film_id, "rating": rate}, 
            ignore_index=True)     

    return df.astype('int64')


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="Sample script for CollaborativeFiltering")

    parse.add_argument('-k', type=int, default=10, help="Neighborhood size. Default 10")
    parse.add_argument('-n', type=int, default=20, help="Items to rate. Default 20")
    parse.add_argument('--rand', action="store_true", help="Generate random ratings")
    parse.add_argument('--mi', type=int, default=10, help="Items to show in recommendation. Default 10")

    args = vars(parse.parse_args())

    # Load data:
    user, items = load_files(ratings='./ml-data/u.data', items='./ml-data/u.item')
    # Create collaborative filter object:
    cf = CollaborativeFilter(user_ratings=user, items_info=items)
    # Get random movies ratings:
    active_user_ratings = get_user_ratings(films=items, user_id=944, n_films=args['n'], rand=args['rand'])
    
    start = time.time()
    # Build the neighborhood:
    neighbor = cf.build_neighborhood(act_user=active_user_ratings, k=args['k'])
    # Compute recommendations and show the title of the movies:
    recomm = cf.recommend(neighborhood=neighbor, act_user=active_user_ratings, min_rating=4, max_items=args['mi'])
    end = time.time()
    
    cf.show(recommendations=recomm)
    print("\nElapsed time: %.2f s" % (end - start))