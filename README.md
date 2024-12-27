# GPU-Tabu-Search

1. Navigate to the program's directory.
2. The instances to be used must be located in the same directory as the program.
3. Compile the program using the command:
   
   nvcc tabu.cu -o executable -O3 -ccbin g++-10

   where tabu.cu is the version to be used (e.g., GPU-TS-f1.cu) and executable is the name for the program's executable.

5. Run the program using the command:
   
   ./executable instance_name dataset_name execution_time_limit number_of_repetitions

6. Example of running the program using the version GPU-TS-f1.cu, the instance MDG-a_21_n2000_m200 from the dataset MDG-a, an execution time limit of 30 seconds, and 3 repetitions:

   nvcc GPU-TS-f1.cu -o tabu_f1 -O3 -ccbin g++-10

   ./tabu_f1 MDG-a_21_n2000_m200 MDG-a 30 3

