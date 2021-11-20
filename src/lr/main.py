from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


from ..data import Data
data=Data()
log_reg = LogisticRegression(solver='lbfgs')

