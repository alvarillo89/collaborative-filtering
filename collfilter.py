# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


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
    

    def __build_item_user_table(self, active):
        # Add active user ratings to the existing ratings:
        users = self.users.append(active, ignore_index=True)
        # Build the item-user table (invested for efficiency):
        table = pd.pivot_table(users, values='rating', index="item_id", 
            columns="user_id", fill_value=0)

        return table
        

    def build_neighborhood(self, act_user, k):
        """ Build the neighborhood from another user ratings:
        act_user -- Pandas dataframe that contains the active user ratings.
            This dataframe must contain the following columns: user_id, item_id, rating.
        k -- size of the neighborhood. Return the k users more similars to active user 
        """
        if not isinstance(act_user, pd.DataFrame):
            raise AttributeError("act_user must be a Pandas DataFrame")
        
        # Init output dataframe:
        df = pd.DataFrame(columns=['user_id', 'Correlation'])        
        # Get active user id:
        active_id = act_user.iloc[0]['user_id']
        table = self.__build_item_user_table(act_user)
        
        # Compute correlations:
        for user in table:
            if user != active_id:
                # Get a tmp dataframe with common items rated:
                tmp = table[[user, active_id]]
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
    

    def __predict(self, weights, ratings):
        """ Predicts expected rating for an item not seen by active user, 
        based on correlation and neighborhood ratings
        """
        wnp = np.array(weights)
        rnp = np.array(ratings)
        num = (wnp * rnp).sum()
        den = np.abs(wnp * np.where(rnp!=0, 1., 0.)).sum()  
        return num / den


    def recommend(self, neighborhood, act_user, min_rating, max_items):
        """ Calculates a dataframe with the expected rating for all items 
        not seen by the active user:
        neighborhood -- Precomputed neighborhood. It must have the same format 
            as the dataframe returned by build_neighborhood method.
        act_user -- Pandas dataframe that contains the active user ratings.
        min_rating -- Items with a predicted rating lower than this parameter 
            will be discarded from the final result.
        max_items -- Maximum number of items to recommend. None to return all 
            recommended items.
        """
        if not (isinstance(neighborhood, pd.DataFrame) and isinstance(act_user, pd.DataFrame)):
            raise AttributeError("neighborhood and act_user must be a Pandas DataFrames")

        # Build table:
        active_id = act_user.iloc[0]['user_id']
        table = self.__build_item_user_table(act_user)

        # Get a reduced version of table, only with the neighborhood users,
        # active user and only rated items:
        columns = neighborhood['user_id'].tolist()
        columns.append(active_id)
        df = table[columns]
        df = df[(df.T != 0).any()]
        # Remove films seen by the active user:
        df = df[(df.T.loc[active_id]==0)]
        # Remove active user:
        df = df.drop(active_id, axis=1)

        # Predict rating for unseen items:
        weights = neighborhood['Correlation'].tolist()
        out = pd.DataFrame(columns=['item_id', 'expected_rating'])
        for index, row in df.iterrows():
            rating = self.__predict(weights, row.tolist())
            out = out.append({'item_id': index, 'expected_rating': rating}, ignore_index=True)
        
        # Prepare the returned df
        out = out.sort_values(by='expected_rating', ascending=False)
        out = out[out['expected_rating'] >= min_rating]
        out = out.astype({"item_id": "int64"}, copy=False)
        out = out.reset_index(drop=True)
        
        return out.head(max_items)


    def show(self, recommendations):
        """ Displays all the information associated to each item on the
        recommendations dataframe
        """
        if not isinstance(recommendations, pd.DataFrame):
            raise AttributeError("recommendations must be a Pandas DataFrame")

        for item in recommendations['item_id'].values:
            value = self.items.loc[self.items['item_id'] == item]
            value = value.drop('item_id', axis=1)
            print(value.iloc[0].tolist())            