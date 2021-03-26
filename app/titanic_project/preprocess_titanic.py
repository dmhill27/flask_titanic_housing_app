import pandas as pd
import string

class PreProcess_Titanic():
  def __init__(self, data):
    self.data = data

  def impute(self, imputer_median, imputer_mode):
    data_num = self.data[self.data.select_dtypes(exclude='object').columns]
    data_cat = self.data[self.data.select_dtypes(include='object').columns]

    num_cols = data_num.columns
    cat_cols = data_cat.columns

    data_num = pd.DataFrame(imputer_median.transform(data_num), columns=num_cols)
    data_cat = pd.DataFrame(imputer_mode.transform(data_cat), columns=cat_cols)

    self.data[num_cols] = data_num
    self.data[cat_cols] = data_cat
  
  def create_cabin_bins(self):
    self.data['Cabin'].fillna('N', inplace=True)
    self.data['Cabin'] = self.data['Cabin'].apply(lambda x: x[0])
  
  def one_hot_encode(self, encoder):
    cat_cols = self.data.select_dtypes(include='object').columns
    data_cat = self.data[cat_cols]
    self.data.drop(columns=cat_cols, inplace=True)
    data_cat_transformed = pd.DataFrame(encoder.transform(data_cat), columns=encoder.get_feature_names(list(cat_cols)))
    self.data = pd.concat([self.data, data_cat_transformed], axis=1)
  
  def create_fare_bins(self):
    fare_bins, fare_labels = [0,15,50,515], [1,2,3]
    self.data['Fare'] = pd.cut(self.data['Fare'], bins=fare_bins, labels=fare_labels, include_lowest=True)
    self.data['Fare'] = self.data['Fare'].astype('float64')
  
  def create_age_bins(self):
    age_bins, age_labels = [0,5,15,60,81], [1,2,3,4]
    self.data['Age'] = pd.cut(self.data['Age'], bins=age_bins, labels=age_labels, include_lowest=True)
    self.data['Age'] = self.data['Age'].astype('float64')
  
  def format_ticket(self):
    def ticket_formatter(ticket):
      ticket = ticket.translate(str.maketrans('', '', string.punctuation))
      if ticket.isalnum():
        prefix = 'None '
        ticket = prefix + str(ticket)
      return ticket

    self.data['Ticket'] = self.data['Ticket'].apply(lambda x: ticket_formatter(x))
    self.data[['TicketPrefix','TicketNumber']] = self.data['Ticket'].str.split(' ', 1, expand=True)
    self.data['TicketNumber'] = self.data['TicketNumber'].apply(lambda x: str(x[0]))
  
  def create_sibsp_bins(self):
    def sibsp_bins(num):
      if num > 2:
        return 'more_than_two'
      elif num > 0:
        return 'one-two'
      else:
        return 'none'
    self.data['SibSp'] = self.data['SibSp'].apply(lambda x: sibsp_bins(x))
  
  def create_parch_bins(self):
    def parch_bins(num):
      if num > 3:
        return 'more_than_three'
      elif num > 0:
        return 'one-three'
      else:
        return 'none'

    
    self.data['Parch'] = self.data['Parch'].apply(lambda x: parch_bins(x))


  def drop_columns(self, cols):
    self.data.drop(columns=cols, inplace=True)

  def return_data(self):
    return self.data