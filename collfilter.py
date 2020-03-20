# -*- coding: utf-8 -*-

import pandas as pd


class CollaborativeFilter:
    """ This class includes methods for building a neighborhood 
    and making recommendations to users based on collaborative 
    filtering 
    """

    def __init__(self, user_ratings, items_info):
        """ Build an object of the class CollaborativeFilter:
        user_ratings -- Pandas dataframe containing the followings columns: 
            user_id, item_id, rating
        items_info -- Pandas dataframe containing at least one column named
            item_id and all the other information that you want to show
            about your items
        """
        if not (isinstance(user_ratings, pd.DataFrame) and isinstance(items_info, pd.DataFrame)):
            raise AttributeError("user_ratings and items_info must be a Pandas DataFrames")
        
        # Store items and user info
        self.items = items_info
        self.users = user_ratings
        

    def build_neighborhood(self, active_user_ratings, k):
        """ Build the neighborhood from another user ratings:
        active_user_ratings -- Pandas dataframe that contains the active user ratings.
            This dataframe must contain the following columns: user_id, item_id, rating.
        k -- size of the neighborhood. Return the k users more similars to active user 
        """
        if not isinstance(active_user_ratings, pd.DataFrame):
            raise AttributeError("active_user_ratings must be a Pandas DataFrame")
        
        # Init output dataframe:
        df = pd.DataFrame(columns=['user_id', 'Correlation'])        

        # Get active user id:
        active_id = active_user_ratings.iloc[0]['user_id']
        # Add active user ratings to the existing ratings:
        self.users = self.users.append(active_user_ratings, ignore_index=True)
        
        # Build the item-user table (invested for efficiency):
        self.table = pd.pivot_table(self.users, values='rating', index="item_id", 
            columns="user_id", fill_value=0)

        # Compute correlations:
        for user in self.table:
            if user != active_id:
                # Get a tmp dataframe with common items rated:
                tmp = self.table[[user, active_id]]
                tmp = tmp[(tmp.T != 0).all()]
                if tmp.shape[0] < 4: continue   # Avoid NaN errors
                corr = pd.DataFrame(tmp.corrwith(tmp[active_id], method='pearson'), 
                    columns=['Correlation'])
                df = df.append({"user_id": corr.index[0], 
                    "Correlation": corr.iloc[0]['Correlation']}, ignore_index=True)

        df = df.astype({"user_id": "int64"}, copy=False)
        df = df.sort_values(by='Correlation', ascending=False).head(k)
        df = df.reset_index(drop=True)
        return df