#!/usr/bin/env python

import zipfile
import shutil
import tempfile
import imp
import numbers
import numpy as np
import sys

test_feature_matrix = np.array([[1,0,0],[0,1,1],[1,0,1],[0,1,0]])
test_labels = np.array([-1,1,-1,1])
test_theta = np.array([1,1,1])
test_theta_0 = 1
test_feature_vector = test_feature_matrix[0]
test_label = test_labels[0]

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
        print student_module
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

def check_hinge_loss(student_module):
    try:
        res = student_module.hinge_loss(test_feature_matrix, test_labels, test_theta, test_theta_0)
        if isinstance(res, numbers.Number):
            print 'hinge_loss: Implemented'
            return True
        else:
            print 'hinge_loss: Expected a numeric return value but got: {0}'.format(res)
            return False
    except NotImplementedError:
        print 'hinge_loss: Not implemented'
        return False
    except:
        print 'hinge_loss: Exception in running hinge_loss'
        return False

def check_perceptron_single_step_update(student_module):
    try:
        res = student_module.perceptron_single_step_update(test_feature_vector, test_label, test_theta, test_theta_0)
        if isinstance(res, tuple):
            print 'perceptron_single_step_update: Implemented'
            return True
        else:
            print 'perceptron_single_step_update: Expected a tuple return value but got: {0}'.format(res)
            return False
    except NotImplementedError:
        print 'perceptron_single_step_update: Not implemented'
        return False
    except:
        print 'perceptron_single_step_update: Exception in running perceptron_single_step_update'
        return False

def check_perceptron(student_module):
    try:
        res = student_module.perceptron(test_feature_matrix, test_labels, 5)
        if isinstance(res, tuple):
            print 'perceptron: Implemented'
            return True
        else:
            print 'perceptron: Expected a tuple return value but got: {0}'.format(res)
            return False
    except NotImplementedError:
        print 'perceptron: Not implemented'
        return False
    except:
        print 'perceptron: Exception in running perceptron'
        return False


def check_passive_aggressive_single_step_update(student_module):
    try:
        res = student_module.passive_aggressive_single_step_update(test_feature_vector, test_label, 1, test_theta, test_theta_0)
        if isinstance(res, tuple):
            print 'passive_aggressive_single_step_update: Implemented'
            return True
        else:
            print 'passive_aggressive_single_step_update: Expected a tuple return value but got: {0}'.format(res)
            return False
    except NotImplementedError:
        print 'passive_aggressive_single_step_update: Not implemented'
        return False
    except:
        print 'passive_aggressive_single_step_update: Exception in running passive_aggressive_single_step_update'
        return False

def check_average_perceptron(student_module):
    try:
        res = student_module.average_perceptron(test_feature_matrix, test_labels, 5)
        if isinstance(res, tuple):
            print 'average_perceptron: Implemented'
            return True
        else:
            print 'average_perceptron: Expected a tuple return value but got: {0}'.format(res)
            return False
    except NotImplementedError:
        print 'average_perceptron: Not implemented'
        return False
    except:
        print 'average_perceptron: Exception in running average_perceptron'
        return False

def check_average_passive_aggressive(student_module):
    try:
        res = student_module.average_passive_aggressive(test_feature_matrix, test_labels, 5, 2)
        if isinstance(res, tuple):
            print 'average_passive_aggressive: Implemented'
            return True
        else:
            print 'average_passive_aggressive: Expected a tuple return value but got: {0}'.format(res)
            return False
    except NotImplementedError:
        print 'average_passive_aggressive: Not implemented'
        return False
    except:
        print 'average_passive_aggressive: Exception in running average_passive_aggressive'
        return False


def check_classify(student_module):
    try:
        res = student_module.classify(test_feature_matrix, test_theta, test_theta_0)
        if isinstance(res, np.ndarray):
            print 'classify: Implemented'
            return True
        else:
            print 'classify: Expected a numpy array return value but got: {0}'.format(res)
            return False
    except NotImplementedError:
        print 'classify: Not implemented'
        return False
    except:
        print 'classify: Exception in running classify'
        return False

def check_perceptron_accuracy(student_module):
    try:
        res = student_module.perceptron_accuracy(test_feature_matrix, test_feature_matrix, test_labels, test_labels, 5)
        if isinstance(res, tuple):
            print 'check_perceptron_accuracy: Implemented'
            return True
        else:
            print 'check_perceptron_accuracy: Expected a tuple return value but got: {0}'.format(res)
            return False
    except NotImplementedError:
        print 'check_perceptron_accuracy: Not implemented'
        return False
    except:
        print 'check_perceptron_accuracy: Exception in running check_perceptron_accuracy'
        return False

def check_average_perceptron_accuracy(student_module):
    try:
        res = student_module.average_perceptron_accuracy(test_feature_matrix, test_feature_matrix, test_labels, test_labels, 5)
        if isinstance(res, tuple):
            print 'average_perceptron_accuracy: Implemented'
            return True
        else:
            print 'average_perceptron_accuracy: Expected a tuple return value but got: {0}'.format(res)
            return False
    except NotImplementedError:
        print 'average_perceptron_accuracy: Not implemented'
        return False
    except:
        print 'average_perceptron_accuracy: Exception in running average_perceptron_accuracy'
        return False

def check_average_passive_aggressive_accuracy(student_module):
    try:
        res = student_module.average_passive_aggressive_accuracy(test_feature_matrix, test_feature_matrix, test_labels, test_labels, 5, 2)
        if isinstance(res, tuple):
            print 'average_passive_aggressive_accuracy: Implemented'
            return True
        else:
            print 'average_passive_aggressive_accuracy: Expected a tuple return value but got: {0}'.format(res)
            return False
    except NotImplementedError:
        print 'average_passive_aggressive_accuracy: Not implemented'
        return False
    except:
        print 'average_passive_aggressive_accuracy: Exception in running average_passive_aggressive_accuracy'
        return False

if __name__ == '__main__':
    zipped_file = 'project1.zip' # name of zip file to be submitted
    required_files = ['main.py', 'project1.py', 'reviews_submit.tsv', 'reviews_test.tsv', 'reviews_train.tsv',
     'reviews_val.tsv', 'stopwords.txt', 'toy_data.tsv', 'utils.py', 'writeup.pdf'] # required files in the zip
    student_file = 'project1.py' # name of student code file
    code_checks = [check_hinge_loss, check_perceptron_single_step_update, check_perceptron,
     check_passive_aggressive_single_step_update, check_average_perceptron, check_average_passive_aggressive, 
     check_classify, check_perceptron_accuracy, check_average_passive_aggressive_accuracy]
    check_zip(zipped_file, required_files, student_file, code_checks)
