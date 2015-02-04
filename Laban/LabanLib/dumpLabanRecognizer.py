import pickle
import LabanUtils.util as labanUtil

X = pickle.load( open( "X", "r" ) )
Y_laban = pickle.load( open( "Y_Laban", "r" ) )
labanClf, selectedIndices = labanUtil.getMultiTaskclassifier(X, Y_laban)

f = open('labanClf', 'w')
f.flush()
pickle.dump(labanClf, f)
f.close()

f = open('selectedIndices', 'w')
f.flush()
pickle.dump(selectedIndices, f)
f.close()
