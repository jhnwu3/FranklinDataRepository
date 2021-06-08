# Set the output to a png file
set terminal png size 800,800
# The file we'll write to
set output 'nonlinearODE.png'
# The graphic title
set title 'Costs within 10% of exact k constants'

# labels
set xlabel "Time (s)"
set ylabel "Concentrations (mM)"
set grid

#plot the graphic
plot "NonlinODE6_Syk_Vav_pVav_SHP1.txt" using 1:2 title "Syk", "NonlinODE6_Syk_Vav_pVav_SHP1.txt" using 1:3 title "Vav", "NonlinODE6_Syk_Vav_pVav_SHP1.txt" using 1:5 title "pVav", "NonlinODE6_Syk_Vav_pVav_SHP1.txt" using 1:6 title "SHP1"

