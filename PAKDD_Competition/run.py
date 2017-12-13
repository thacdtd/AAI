#!/usr/bin/env python

#############################
# ChaLearn AutoML2 challenge #
#############################

# Usage: python program_dir/run.py input_dir output_dir program_dir

# program_dir is the directory of this program

#
# The input directory input_dir contains 5 subdirectories named by dataset,
# including:
# 	dataname/dataname_feat.type          -- the feature type "Numerical", "Binary", or "Categorical" (Note: if this file is abscent, get the feature type from the dataname.info file)
# 	dataname/dataname_public.info        -- parameters of the data and task, including metric and time_budget
# 	dataname/dataname_test.data          -- training, validation and test data (solutions/target values are given for training data only)
# 	dataname/dataname_train.data
# 	dataname/dataname_train.solution
# 	dataname/dataname_valid.data
#
# The output directory will receive the predicted values (no subdirectories):
# 	dataname_valid.predict           
# 	dataname_test.predict
# We have 2 test sets named "valid" and "test", please provide predictions for both.
# 
# We implemented 2 classes:
#
# 1) DATA LOADING:
#    ------------
# Use/modify 
#                  D = DataManager(basename, input_dir, ...) 
# to load and preprocess data.
#     Missing values --
#       Our default method for replacing missing values is trivial: they are replaced by 0.
#       We also add extra indicator features where missing values occurred. This doubles the number of features.
#     Categorical variables --
#       The location of potential Categorical variable is indicated in D.feat_type.
#       NOTHING special is done about them in this sample code. 
#     Feature selection --
#       We only implemented an ad hoc feature selection filter efficient for the 
#       dorothea dataset to show that performance improves significantly 
#       with that filter. It takes effect only for binary classification problems with sparse
#       matrices as input and unbalanced classes.
#
# 2) LEARNING MACHINE:
#    ----------------
# Use/modify 
#                 M = MyAutoML(D.info, ...) 
# to create a model.
#     Number of base estimators --
#       Our models are ensembles. Adding more estimators may improve their accuracy.
#       Use M.model.n_estimators = num
#     Training --
#       M.fit(D.data['X_train'], D.data['Y_train'])
#       Fit the parameters and hyper-parameters (all inclusive!)
#       What we implemented hard-codes hyper-parameters, you probably want to
#       optimize them. Also, we made a somewhat arbitrary choice of models in
#       for the various types of data, just to give some baseline results.
#       You probably want to do better model selection and/or add your own models.
#     Testing --
#       Y_valid = M.predict(D.data['X_valid'])
#       Y_test = M.predict(D.data['X_test']) 
#
# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 
#
# Main contributors: Isabelle Guyon and Arthur Pesah, March-October 2014
# Lukasz Romaszko April 2015
# Originally inspired by code code: Ben Hamner, Kaggle, March 2013
# Modified by Ivan Judson and Christophe Poulain, Microsoft, December 2013
# Last modifications Isabelle Guyon, November 2017

# =========================== BEGIN USER OPTIONS ==============================
# Verbose mode: 
##############
# Recommended to keep verbose = True: shows various progression messages
verbose = True  # outputs messages to stdout and stderr for debug purposes

# Debug level:
############## 
# 0: run the code normally, using the time budget of the tasks
# 1: run the code normally, but limits the time to max_time
# 2: run everything, but do not train, generate random outputs in max_time
# 3: stop before the loop on datasets
# 4: just list the directories and program version
debug_mode = 0

# Time budget
#############
# Maximum time of training in seconds PER DATASET (there are 5 datasets). 
# The code should keep track of time spent and NOT exceed the time limit 
# in the dataset "info" file, stored in D.info['time_budget'], see code below.
# If debug >=1, you can decrease the maximum time (in sec) with this variable:
max_time = 1200

# Maximum number of cycles, number of samples, and estimators
#############################################################
# Your training algorithm may be fast, so you may want to limit anyways the 
# number of points on your learning curve (this is on a log scale, so each 
# point uses twice as many time than the previous one.)
# The original code was modified to do only a small "time probing" followed
# by one single cycle. We can now also give a maximum number of estimators 
# (base learners).
max_cycle = 1
max_estimators = 10
max_samples = float('Inf')

# I/O defaults
##############
# If true, the previous output directory is not overwritten, it changes name
save_previous_results = False
# Use default location for the input and output data:
# If no arguments to run.py are provided, this is where the data will be found
# and the results written to. Change the root_dir to your local directory.
root_dir = ""
default_input_dir = root_dir + "sample_data"
default_output_dir = root_dir + "AutoML2_sample_result_submission"
default_program_dir = root_dir + "AutoML2_sample_code_program"

# =============================================================================
# =========================== END USER OPTIONS ================================
# =============================================================================

# Version of the sample code
version = 5

# General purpose functions
import time
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score

overall_start = time.time()  # <== Mark starting time
import os
from sys import argv, path
import datetime

the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

# =========================== BEGIN PROGRAM ================================

if __name__ == "__main__" and debug_mode < 4:
    #### Check whether everything went well (no time exceeded)
    execution_success = True

    #### INPUT/OUTPUT: Get input and output directory names
    if len(argv) == 1:  # Use the default input and output directories if no arguments are provided
        input_dir = default_input_dir
        output_dir = default_output_dir
        program_dir = default_program_dir
    else:
        input_dir = os.path.abspath(argv[1])
        output_dir = os.path.abspath(argv[2])
        program_dir = os.path.abspath(argv[3])

    if verbose:
        print("Using input_dir: " + input_dir)
        print("Using output_dir: " + output_dir)
        print("Using program_dir: " + program_dir)

    # Our libraries
    path.append(program_dir + "/lib/")
    path.append(input_dir)
    import lib.data_io as data_io  # general purpose input/output functions
    from lib.data_io import vprint  # print only in verbose mode
    from lib.data_manager import DataManager  # load/save data and get info about them
    from lib.models import MyAutoML  # example model

    if debug_mode >= 4:  # Show library version and directory structure
        data_io.show_dir(".")

    # Move old results and create a new output directory (useful if you run locally)
    if save_previous_results:
        data_io.mvdir(output_dir, output_dir + '_' + the_date)
    data_io.mkdir(output_dir)

    #### INVENTORY DATA (and sort dataset names alphabetically)
    datanames = data_io.inventory_data(input_dir)
    # Overwrite the "natural" order

    #### DEBUG MODE: Show dataset list and STOP
    if debug_mode >= 3:
        data_io.show_version()
        data_io.show_io(input_dir, output_dir)
        print('\n****** Ingestion program version ' + str(version) + ' ******\n\n' + '========== DATASETS ==========\n')
        data_io.write_list(datanames)
        datanames = []  # Do not proceed with learning and testing

    #### MAIN LOOP OVER DATASETS: 
    overall_time_budget = 0
    time_left_over = 0
    for basename in datanames:  # Loop over datasets

        vprint(verbose, "\n========== Ingestion program version " + str(version) + " ==========\n")
        vprint(verbose, "************************************************")
        vprint(verbose, "******** Processing dataset " + basename.capitalize() + " ********")
        vprint(verbose, "************************************************")

        # ======== Learning on a time budget:
        # Keep track of time not to exceed your time budget. Time spent to inventory data neglected.
        start = time.time()

        # ======== Creating a data object with data, informations about it
        vprint(verbose, "========= Reading and converting data ==========")
        D = DataManager(basename, input_dir, replace_missing=True, filter_features=True, max_samples=max_samples,
                        verbose=verbose)
        print D
        vprint(verbose, "[+] Size of uploaded data  %5.2f bytes" % data_io.total_size(D))

        # ======== Keeping track of time
        if debug_mode < 1:
            time_budget = D.info['time_budget']  # <== HERE IS THE TIME BUDGET!
        else:
            time_budget = max_time
        # print overall_time_budget
        # print time_budget
        time_budget = float(time_budget)
        overall_time_budget = overall_time_budget + time_budget
        vprint(verbose, "[+] Cumulated time budget (all tasks so far)  %5.2f sec" % (overall_time_budget))
        # We do not add the time left over form previous dataset: time_budget += time_left_over
        vprint(verbose, "[+] Time budget for this task %5.2f sec" % time_budget)
        time_spent = time.time() - start
        vprint(verbose, "[+] Remaining time after reading data %5.2f sec" % (time_budget - time_spent))
        if time_spent >= time_budget:
            vprint(verbose, "[-] Sorry, time budget exceeded, skipping this task")
            execution_success = False
            continue

        # ========= Creating a model, knowing its assigned task from D.info['task'].
        # The model can also select its hyper-parameters based on other elements of info.  
        vprint(verbose, "======== Creating model ==========")
        M = MyAutoML(D.info, verbose=False, debug_mode=debug_mode)  # I turned off verbose to avoid tons of junk...
        print M

        # ========= Iterating over learning cycles and keeping track of time
        time_spent = time.time() - start
        vprint(verbose, "[+] Remaining time after building model %5.2f sec" % (time_budget - time_spent))
        if time_spent >= time_budget:
            vprint(verbose, "[-] Sorry, time budget exceeded, skipping this task")
            execution_success = False
            continue

        time_budget = time_budget - time_spent  # Remove time spent so far
        start = time.time()  # Reset the counter
        time_spent = 0  # Initialize time spent learning
        cycle = 0

        while time_spent <= time_budget / 2 and cycle <= max_cycle and M.model.n_estimators < max_estimators:
            vprint(verbose,
                   "=========== " + basename.capitalize() + " Training cycle " + str(cycle) + " ================")
            # Estimate the number of base estimators
            # --------------------------------------
            if cycle == 1 and max_cycle == 1:
                # Directly use up all time left in one iteration
                n_estimators = M.model.n_estimators
                new_n_estimators = int((np.floor(time_left_over / time_spent) - 1) * n_estimators)
                if new_n_estimators <= n_estimators: break
                M.model.n_estimators = new_n_estimators
            else:
                # Make a learning curve by exponentially increasing the number of estimators
                M.model.n_estimators = int(np.exp2(cycle))
            M.model.n_estimators = min(max_estimators, M.model.n_estimators)
            vprint(verbose, "[+] Number of estimators: %d" % (M.model.n_estimators))
            # Fit base estimators
            # -------------------
            M.fit(D.data['X_train'], D.data['Y_train'])
            vprint(verbose, "[+] Fitting success, time spent so far %5.2f sec" % (time.time() - start))
            vprint(verbose, "[+] Size of trained model  %5.2f bytes" % data_io.total_size(M))
            # Make predictions
            # -----------------
            Y_valid = M.predict(D.data['X_valid'])
            Y_test = M.predict(D.data['X_test'])
            # print("???????????????????")
            # print Y_valid
            # print Y_test
            # print roc_auc_score(Y_valid, Y_test)
            # print("???????????????????")
            vprint(verbose, "[+] Prediction success, time spent so far %5.2f sec" % (time.time() - start))
            # Write results
            # -------------
            filename_valid = basename + '_valid.predict'
            filename_test = basename + '_test.predict'
            data_io.write(os.path.join(output_dir, filename_valid), Y_valid)
            data_io.write(os.path.join(output_dir, filename_test), Y_test)
            vprint(verbose, "[+] Results saved, time spent so far %5.2f sec" % (time.time() - start))
            time_spent = time.time() - start
            time_left_over = time_budget - time_spent
            vprint(verbose, "[+] End cycle, time left %5.2f sec" % time_left_over)
            if time_left_over <= 0: break
            cycle += 1

    overall_time_spent = time.time() - overall_start
    if execution_success:
        vprint(verbose, "[+] Done")
        vprint(verbose,
               "[+] Overall time spent %5.2f sec " % overall_time_spent + "::  Overall time budget %5.2f sec" % overall_time_budget)
    else:
        vprint(verbose, "[-] Done, but some tasks aborted because time limit exceeded")
        vprint(verbose,
               "[-] Overall time spent %5.2f sec " % overall_time_spent + " > Overall time budget %5.2f sec" % overall_time_budget)
