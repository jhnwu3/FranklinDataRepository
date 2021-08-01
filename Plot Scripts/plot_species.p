# Set the output to a png file
set terminal png size 800,800
# The file we'll write to
set output 'nonlinearODE.png'
# The graphic title
set title 'Nonlinear ODE System'

# labels
set xlabel "Time (s)"
set ylabel "Concentrations (mM)"
set grid

#plot the graphic
plot "Nonlin_ODE_6_Plot.txt" using 1:2 title "Syk", "Nonlin_ODE_6_Plot.txt" using 1:3 title "Vav", "Nonlin_ODE_6_Plot.txt" using 1:5 title "pVav", "Nonlin_ODE_6_Plot.txt" using 1:6 title "SHP1"

