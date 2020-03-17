# -*- coding: utf-8 -*-

import pandas as pd


def load_files(ratings, items):
    """ This is just a example load function, made for the MovieLens
    collection, you can write your own for your own collection 
    """

    user_names = ("user_id", "item_id", "rating", "timestamp")
    items_names = ("movie_id","movie_title","release_date","video_release_date",
        "IMDb_URL","unknown","Action","Adventure","Animation",
        "Children's","Comedy","Crime","Documentary","Drama","Fantasy",
        "Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi",
        "Thriller","War","Western")

    user = pd.read_csv(ratings, sep='\t', names=user_names, encoding="ISO-8859-1")
    items = pd.read_csv(items, sep='|', names=items_names, encoding="ISO-8859-1")

    return user, items


class CollaborativeFilter:
    """ This class includes methods for building a neighborhood 
    and making recommendations to users based on collaborative 
    filtering 
    """

    def __init__(self, user_ratings, items_info):
        """ Build an object of the class CollaborativeFilter:
        user_ratings -- Pandas dataframe containing the rating that the users 
            made of the items
        items_info -- Pandas dataframe containing the items info. Each rating must
            be related with some item of this file by its ID.
        """
        if not (isinstance(user_ratings, pd.DataFrame) and isinstance(items_info, pd.DataFrame)):
            raise AttributeError("user_ratings and items_info must be Pandas DataFrames")
        
        self.user = user_ratings
        self.items = items_info
       
    
    def build_neighborhood(self, active_user_ratings):
        pass