# -*- coding: utf-8 -*-

import random
import pandas as pd
from collfilter import CollaborativeFilter, load_files


def get_user_ratings(films, n_films=10):
    """ Ask active user to rate, from 1 to 5, `n_films` random films from
    `films` DataFrame. Returns a new dataframe with film id and user rating
    """
    n = films.shape[0]
    df = pd.DataFrame(columns=("item_id", "rating"))
    for _ in range(n_films):
        index = random.randint(0, n-1)
        film_title = films.iloc[index]["movie_title"]
        film_id = films.iloc[index]["movie_id"]
        rate = input("What do you think about %s? [1-5]: " % film_title)
        
        while type(rate) is str:
            try:
                rate = int(rate)
                if not (1 <= rate and rate <= 5): raise Exception
            except Exception:
                rate = input("Oops! Something went wrong, ensure that your" + 
                    "rate is between 1 and 5: ")

        df = df.append({'item_id': film_id, "rating": rate}, ignore_index=True)     
    
    return df


if __name__ == "__main__":
    # Load data:
    user, items = load_files(ratings='./ml-data/u.data', items='./ml-data/u.item')
    cf = CollaborativeFilter(user_ratings=user, items_info=items)
    active_user_ratings = get_user_ratings(items)
