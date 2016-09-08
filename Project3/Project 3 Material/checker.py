import zipfile
import shutil
import tempfile
import imp
import numbers
import numpy as np
import sys

def check_zip(zipped_file, required_files, student_file, code_checks):
    """ Checks whether files in required_files are present in the zipped_file and basic code behavior """

    f = open(zipped_file, 'rb')
    z = zipfile.ZipFile(f)
    file_list = z.namelist()
    files_not_found = []
    for filename in required_files:
        if filename not in file_list:
            files_not_found.append(filename)

    if files_not_found:
        f.close()
        print 'The following files are missing: {0}'.format(', '.join(files_not_found))
        return False

    print 'All required files present'

    # extract the zip to a temporary directory and check basic behavior
    tempdir = tempfile.mkdtemp()
    z.extractall(tempdir)
    try:
        sys.path[0] = tempdir
        student_module = imp.load_source('student_code', tempdir + '/' + student_file)
    except Exception, e: 
        shutil.rmtree(tempdir)
        f.close()
        print 'Error in importing your code'
        return False

    for check_fn in code_checks:
        check_fn(student_module)

    # delete the temporary directory
    shutil.rmtree(tempdir)
    f.close()

    return True

def load_test1():
    global test_n,test_d,test_K,test_X,test_Mu,test_P,test_Var,test_post
    test_n=10
    test_d=20
    test_K =5
    test_X = np.reshape(np.ceil(5*np.random.uniform(0,1,test_d*test_n)),(test_n,test_d))
    test_Mu= np.reshape(np.ceil(5*np.random.uniform(0,1,test_K*test_d)),(test_K,test_d))
    test_P= np.tile(1./test_K,test_K)
    test_Var=np.tile(1,test_K)
    test_post=np.tile(1./test_K,(test_n,test_K))
def load_test2(student_module):
	global test_n,test_d,test_K,test_X,test_Mu,test_P,test_Var,test_post
	test_X = student_module.readData('toy_data.txt')
	(test_Mu,test_P,test_Var) = student_module.init(test_X,test_K)

def check_Estep(student_module):
	load_test1()
	try:
		res = student_module.Estep(test_X, test_K, test_Mu, test_P,test_Var)
		if not isinstance(res, tuple):
			print 'Estep: Expected a tuple return value but got: {0}'.format(res)
			return False
		elif not isinstance(res[0],np.ndarray):
			print 'Estep: Expected a numpy array return value for the first output but got: {0}'.format(res[0])
			return False
		elif not isinstance(res[1],numbers.Number):
			print 'Estep: Expected a numeric return value for the second output but got: {0}'.format(res[1])
			return False
		else:
			print 'Estep: Type checked'
			return True
	except NotImplementedError:
		print 'Estep: Not implemented'
		return False
	except:
		print 'Estep: Exception in running Estep'
		return False

def check_Estep_part2(student_module):
	load_test1()
	try:
		res = student_module.Estep_part2(test_X, test_K, test_Mu, test_P,test_Var)
		if not isinstance(res, tuple):
			print 'Estep_part2: Expected a tuple return value but got: {0}'.format(res)
			return False
		elif not isinstance(res[0],np.ndarray):
			print 'Estep_part2: Expected a numpy array return value for the first output but got: {0}'.format(res[0])
			return False
		elif not isinstance(res[1],numbers.Number):
			print 'Estep_part2: Expected a numeric return value for the second output but got: {0}'.format(res[1])
			return False
		else:
			print 'Estep_part2: Type checked'
			return True
	except NotImplementedError:
		print 'Estep_part2: Not implemented'
		return False
	except:
		print 'Estep_part2: Exception in running Estep_part2'
		return False

def check_Mstep(student_module):
	load_test1()
	try:
		res = student_module.Mstep(test_X, test_K, test_Mu, test_P,test_Var,test_post)
		if not isinstance(res, tuple):
			print 'Mstep: Expected a tuple return value but got: {0}'.format(res)
			return False
		elif not isinstance(res[0],np.ndarray):
			print 'Mstep: Expected a numpy array return value for the first output but got: {0}'.format(res)
			return False
		elif not isinstance(res[1],np.ndarray):
			print 'Mstep: Expected a numpy array return value for the second output but got: {0}'.format(res)
			return False
		elif not isinstance(res[2],np.ndarray):
			print 'Mstep: Expected a numpy array return value for the third output but got: {0}'.format(res)
			return False
		else:
			print 'Mstep: Type checked'
			return True
	except NotImplementedError:
		print 'Mstep: Not implemented'
		return False
	except:
		print 'Mstep: Exception in running Mstep'
		return False

def check_Mstep_part2(student_module):
	load_test1()
	try:
		res = student_module.Mstep_part2(test_X, test_K, test_Mu, test_P,test_Var,test_post)
		if not isinstance(res, tuple):
			print 'Mstep_part2: Expected a tuple return value but got: {0}'.format(res)
			return False
		elif not isinstance(res[0],np.ndarray):
			print 'Mstep_part2: Expected a numpy array return value for the first output but got: {0}'.format(res)
			return False
		elif not isinstance(res[1],np.ndarray):
			print 'Mstep_part2: Expected a numpy array return value for the second output but got: {0}'.format(res)
			return False
		elif not isinstance(res[2],np.ndarray):
			print 'Mstep_part2: Expected a numpy array return value for the third output but got: {0}'.format(res)
			return False
		else:
			print 'Mstep_part2: Type checked'
			return True
	except NotImplementedError:
		print 'Mstep_part2: Not implemented'
		return False
	except:
		print 'Mstep_part2: Exception in running Mstep_part2'
		return False

def check_mixGauss(student_module):
	load_test2(student_module)
	try:
		res = student_module.mixGauss(test_X, test_K, test_Mu, test_P,test_Var)
		if not isinstance(res, tuple):
			print 'mixGauss: Expected a tuple return value but got: {0}'.format(res)
			return False
		elif not isinstance(res[0],np.ndarray):
			print 'mixGauss: Expected a numpy array return value for the first output but got: {0}'.format(res)
			return False
		elif not isinstance(res[1],np.ndarray):
			print 'mixGauss: Expected a numpy array return value for the second output but got: {0}'.format(res)
			return False
		elif not isinstance(res[2],np.ndarray):
			print 'mixGauss: Expected a numpy array return value for the third output but got: {0}'.format(res)
			return False
		elif not isinstance(res[3],np.ndarray):
			print 'mixGauss: Expected a numpy array return value for the forth output but got: {0}'.format(res)
			return False
		elif not isinstance(res[4],np.ndarray):
			print 'mixGauss: Expected a numpy array return value for the fifth output but got: {0}'.format(res)
			return False
		else:
			print 'mixGauss: Type checked'
			return True
	except NotImplementedError:
		print 'mixGauss: Not implemented'
		return False
	except:
		print 'mixGauss: Exception in running mixGauss'
		return False

def check_mixGauss_part2(student_module):
	load_test2(student_module)
	try:
		res = student_module.mixGauss_part2(test_X, test_K, test_Mu, test_P,test_Var)
		if not isinstance(res, tuple):
			print 'mixGauss_part2: Expected a tuple return value but got: {0}'.format(res)
			return False
		elif not isinstance(res[0],np.ndarray):
			print 'mixGauss_part2: Expected a numpy array return value for the first output but got: {0}'.format(res)
			return False
		elif not isinstance(res[1],np.ndarray):
			print 'mixGauss_part2: Expected a numpy array return value for the second output but got: {0}'.format(res)
			return False
		elif not isinstance(res[2],np.ndarray):
			print 'mixGauss_part2: Expected a numpy array return value for the third output but got: {0}'.format(res)
			return False
		elif not isinstance(res[3],np.ndarray):
			print 'mixGauss_part2: Expected a numpy array return value for the forth output but got: {0}'.format(res)
			return False
		elif not isinstance(res[4],np.ndarray):
			print 'mixGauss_part2: Expected a numpy array return value for the fifth output but got: {0}'.format(res)
			return False
		else:
			print 'mixGauss_part2: Type checked'
			return True
	except NotImplementedError:
		print 'mixGauss_part2: Not implemented'
		return False
	except:
		print 'mixGauss_part2: Exception in running mixGauss_part2'
		return False

def check_BICmix(student_module):
	load_test2(student_module)
	try:
		res = student_module.BICmix(test_X, [1,2,3])
		if not isinstance(res, numbers.Number):
			print 'BICmix: Expected a numeric return value but got: {0}'.format(res)
		else:
			print 'BICmix: Type checked'
			return True
	except NotImplementedError:
		print 'BICmix: Not implemented'
		return False
	except:
		print 'BICmix: Exception in running BICmix'
		return False

def check_fillMatrix(student_module):
	load_test2(student_module)
	try:
		res = student_module.fillMatrix(test_X, test_K, test_Mu, test_P, test_Var)
		if not isinstance(res, np.ndarray):
			print 'fillMatrix: Expected a numpy array return value but got: {0}'.format(res)
		else:
			print 'fillMatrix: Type checked'
			return True
	except NotImplementedError:
		print 'fillMatrix: Not implemented'
		return False
	except:
		print 'fillMatrix: Exception in running fillMatrix'
		return False

if __name__ == '__main__':
    zipped_file = 'project3.zip' # name of zip file to be submitted
    required_files = ['main.py', 'project3.py'] # required files in the zip
    student_file = 'project3.py' # name of student code file
    code_checks = [check_Estep,check_Mstep,check_mixGauss,check_BICmix,check_Estep_part2,check_Mstep_part2,check_mixGauss_part2,check_fillMatrix]
    check_zip(zipped_file, required_files, student_file, code_checks)