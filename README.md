# Programmability Evaluation Repo Instructions

0) Download your key file here: `<KEYFILE.pem>`

1) Access your virtual machine by running
```
ssh -i <KEYFILE.pem> ubuntu@<SYSTEM_IP>
```
Notes:
 - VMs may take a couple of minutes to start
 - If you get a warning about permissions, you may have to run `chmod 600 <KEYFILE.pem>`
 - If you're using Windows, you may have to convert the key to `.ppk`
 
2) Once you log into your VM, run 
```
sh startEval.sh
```
to enter a Docker container that contains the system documented below.  __If you forget to do this, you will run into lots of 

3) Navigate to `/data/test/{task}` according to the task you've been assigned.

The `/data/test/{task}` directory contains a number of files:
```
├── README.md ──────── documentation
├── docs ───────────── more documentation (figures, etc)
│
│── run.sh ─────────── script showing how code should be run
├── prep.py ────────── code for dataset preprocessing
│                      Note: The data is already preprocessed, so _you do not need to run this script_
│
├── data ───────────── data (already preprocessed)
│
│── main-redacted.py ─ implementation skeleton
│── logs ───────────── verbose logs from a reference implementation to aid testers.
└── validate.py ────── validation script
```

You should start by reading the `README.md` and any associated materials.

__The testers' task is to write a functional `main.py` that passes the correctness check in `validate.py`__.  `main-redacted.py` is an implementation skeleton that illustrates how IO should work and specifies all of the necessary parameters.  You do not _have_ to use any code from `main-redacted.py` in your implementation, but you should make sure that results are written in the _exact_ same format and location, or the correctness check in `validate.py` will fail.  __`main-redacted.py` is intended to rough out one way that a correct solution could be implemented.  Note that some of the functions referenced in `main-redacted.py` may not be available in the performer system you're testing.__

You do not have to run `prep.py` -- preprocessed data can be found in the `data` directory.  We've included `prep.py` to provide insight into the format of the preprocessed data, which may or may not be helpful.

There are verbose logs from the reference implementation contained in the `logs` directory.  These are included as a debugging aid for testers.  Some workflows (`lgc` and `ipnsw`) run the same function on multiple inputs -- in this case, the logs only contain output for the first input.  For neural network workflows, we just show a trace of the first training iteration.

Note that the `validate.py` scripts require `python>=3.6` and may require some common packages (`numpy` and `scipy`).  This should not be very sensitive to versions, but the versions that the tests were written with are shown in `install.sh`.

4) Implement the algorithm!  As shown in `run.sh`, use `validate.py` to make sure that your results are correct.  

5) Once you have a correct implementation, try to improve the performance of your implementation using the profiling tools provided with your specific system.

6) __Important__  When you are done:
 - your __must write your results to the location specified in `main-redacted.py`__ and they __must pass the correctness check in `validate.py`__ or your submission will be considered incomplete.
 - The specific system you're testing has some way to measure/estimate the performance of your code.  __You must save this output to a file called `/data/test/{task}/profiling_results.txt`__ or your submission will be considered incomplete.

