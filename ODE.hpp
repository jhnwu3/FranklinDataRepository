#include "main.hpp"
/* ODE Systems Functions */
void nonlinearODE3( const state_type &c , state_type &dcdt , double t );
void linearODE3_true( const state_type &c , state_type &dcdt , double t );
void linearODEn_1( const state_type &c , state_type &dcdt , double t );
void nonlinearODE6( const state_type &c , state_type &dcdt , double t);