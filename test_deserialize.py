

import pandas
import deserialize_class


model_file_name ="weights.h5"
data_file_name = "test_set.tabular"

deserialize = deserialize_class.DeserializeClass(model_file_name)
classifier = deserialize.deserialize_class()
print(classifier)
'''data = pandas.read_csv(data_file_name, sep='\t', header='infer', index_col=None, parse_dates=True, encoding=None, tupleize_cols=False)
prediction = classifier.predict(data)
print(prediction)'''
#prediction.to_csv(path_or_buf = "pred.csv", sep="\t", index=False)
