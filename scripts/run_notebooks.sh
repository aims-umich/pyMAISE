# Notebooks
declare -a notebooks=(
  "mit_reactor"
  "reactor_physics"
  "fuel_performance"
  "heat_conduction"
  "bwr"
  "HTGR_microreactor"
  "rod_ejection"
  "chf"
)

for notebook in "${notebooks[@]}"
do
  jupyter nbconvert --execute --to notebook --inplace ../docs/source/benchmarks/${notebook}.ipynb
done