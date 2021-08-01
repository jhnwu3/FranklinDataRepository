# Set the output to a png file
set terminal png size 800,800
# The file we'll write to
set output 'nonlinCosts.png'
# The graphic title
set title 'Nonlinear Costs'

# labels
set xlabel "Iteration"
set ylabel "NL Cost"
set grid

#plot the graphic
plot "<(sed -n '1,10p' Jay_Global_Best.txt)"  using 1:8 title "NL Cost" with points pointtype 5

