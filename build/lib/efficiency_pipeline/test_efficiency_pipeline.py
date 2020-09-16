##Test File
import sys,os,traceback
from copy import deepcopy
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import efficiency_pipeline

def test_efficiency_pipeline():
	failed=0
	total=0
	#Fill in tests here.
	try:   
		total+=1 
		print('Testing package...',end='')
		print('Passed!')
	except Exception as e:
		print('Failed')
		print(traceback.format_exc())
		
		failed+=1

	print('Passed %i/%i tests.'%(total-failed,total))

	return

if __name__ == '__main__':
	test_efficiency_pipeline()
