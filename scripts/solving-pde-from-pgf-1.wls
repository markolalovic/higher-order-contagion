#!/usr/local/bin/wolframscript

(* solving-pde-from-pgf.wls 

Got a warning:

```mathematica
NDSolve::bcart: Warning: an insufficient number of boundary conditions have been 
specified for the direction of independent variable s. 
Artificial boundary effects may be present in the solution.
```

But solution seems to be correct:
```mathematica
Check sum p_k over all k at t_max / 2 is 1:
g(1, tMax/2) = 1.
```

And it matches solution from `p(t) = e^{Q t} p0` (see: `expected-values-varying-N-1.pdf`).

*)

ClearAll["Global`*"]
(* :!CodeAnalysis::BeginBlock:: *)
(* :!CodeAnalysis::Disable::SuspiciousSessionSymbol:: *)

figurePath = "../figures/solutions-pde-from-pgf/solving-pde-from-pgf-1.pdf";

(* --- Setup --- *)
Nval = 6;   (* number of nodes *)
I0 = 1;     (* initial infected *)

beta1 = 2.0 / Nval; 
beta2 = 4.0 / Nval^2;
mu = 1;

tMax = 10;

Print["N = ", Nval, ", I0 = ", I0, " beta1 = ", beta1, ", beta2 = ", beta2, ", mu = ", mu, ", tMax = ", tMax];

(* ----------------------------------------------------- *)
(*  PDE from PGF                                         *)
(* ----------------------------------------------------- *)
pde = D[g[s, t], t] == (s - 1) * (
     beta1 * ( (Nval - 1) * s * D[g[s, t], s] - s^2 * D[g[s, t], {s, 2}] ) +
     (beta2 / 2) * ( (Nval - 2) * s^2 * D[g[s, t], {s, 2}] - s^3 * D[g[s, t], {s, 3}] ) - 
     mu * D[g[s, t], s]
);

(* initial condition and boundary condition *)
ic = g[s, 0] == s^I0;
bc = g[1, t] == 1;

(* Solve the PDE *)
pgfSol = NDSolve[{pde, ic, bc}, g, {t, 0, tMax}, {s, 0, 1}];
gSol = g /. First[pgfSol];
Print["Check sum p_k over all k at t_max / 2 is 1:"]
Print["g(1, tMax/2) = ", gSol[1, tMax / 2]]

(* --- Extract expected values from PGF solution --- *)
dgds[s_, t_] = D[gSol[s, t], s];
expectedValueFunction[t_] := dgds[1, t];

(* And plot them *)
pltLabel = Row[{"Expected values from PGF, with N = ", Nval, ", I0 = ", I0, ", mu = 1, beta1 = 2 / N, beta2 = 4 / N^2"}];
plotExpValFromPGF = Plot[expectedValueFunction[t], {t, 0, tMax},
  AxesLabel -> {"Time t", "E[X(t)]"},
  PlotLabel -> pltLabel,
  GridLines -> Automatic,
  ImageSize -> Large,
  PlotRange -> {0, 1.6} (* y-axis should start at 0 *)  
];
(* plotExpValFromPGF *)
Export[figurePath, plotExpValFromPGF];

Print["Done, saved to: ", figurePath];
(* :!CodeAnalysis::EndBlock:: *)