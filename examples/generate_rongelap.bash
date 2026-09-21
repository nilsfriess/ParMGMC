export IDX=4
python rongelap.py -snes_type poissongibbs --idx ${IDX} --output results_poissongibbs.txt
python rongelap.py -snes_type poissongibbsfas --idx ${IDX} --output results_poissongibbsfas.txt
python rongelap.py -snes_type poissonmala -poissonmala_stepsize 0.8 --idx ${IDX} --output results_poissonmala.txt
sed -n "$((IDX+1))p" ../data/rate_samples.txt > results_vangelis.txt
sed -i 's/ /,/g' results_vangelis.txt