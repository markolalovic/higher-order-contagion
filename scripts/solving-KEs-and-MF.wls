#!/usr/local/bin/wolframscript
ClearAll["Global`*"]
figurePath = "../figures/solutions_pde_from_pgf/solutions_KEs_MF.pdf";

(* --- Setup --- *)
Nval = 1000;   (* number of nodes *)
I0 = 50;       (* initial infected *)

(* Rates *)
beta1Param = 2.4 / Nval;
beta2Param = 4.4 / Nval^2;
muParam = 1;

tMax = 10.0;
Print["Setup: "]
Print["N = ", Nval, ", I0 = ", I0, ", beta1 = ", beta1Param, ", beta2 = ", beta2Param, ", mu = ", muParam];

(* ----------------------------------------------------- *)
(*  Solve Kolmogorov Equations                           *)
(* ----------------------------------------------------- *)
(* Exact rates for complete simplicial complex *)
a[k_] := beta1Param * k * (Nval - k);
b[k_] := beta2Param * Binomial[k, 2] * (Nval - k);
c[k_] := muParam * k;

(* System of ODEs fwd KEs *)
equations = Table[
    p[k]'[t] == 
        If[k > 0, (a[k-1] + b[k-1]) * p[k-1][t], 0] -
        (a[k] + b[k] + c[k]) * p[k][t] + 
        If[k < Nval, c[k+1] * p[k+1][t], 0],
    {k, 0, Nval}
];

(* Initial conditions *)
initialConditions = Table[
    p[k][0] == If[k == I0, 1, 0],
    {k, 0, Nval}
];

(* Solve the system with stiff solver-  Handle stiffness *)
Print["Solving Kolmogorov equations with stiff solver..."];
sol = NDSolve[
    Join[equations, initialConditions],
    Table[p[k], {k, 0, Nval}],
    {t, 0, tMax},
    Method -> "StiffnessSwitching",
    AccuracyGoal -> 8,
    PrecisionGoal -> 8,
    MaxSteps -> Infinity
];

(* Extract expected value *)
expectedValueKE[t_] := Sum[k * p[k][t], {k, 0, Nval}] /. sol[[1]];


(* ----------------------------------------------------- *)
(*  Mean Field Moment - based approach                   *)
(* ----------------------------------------------------- *)
meanFieldEq = {
    y'[t] == beta1Param * y[t] * (Nval - y[t]) + 
             beta2Param * y[t] * (y[t] - 1) * (Nval - y[t])/2 - 
             muParam * y[t],
    y[0] == I0
};

mfSol = NDSolve[meanFieldEq, y, {t, 0, tMax}];
expectedValueMF[t_] := y[t] /. mfSol[[1]];


(* ----------------------------------------------------- *)
(*  Plot comparison                                     *)
(* ----------------------------------------------------- *)
pltLabel = Row[{"Expected values for N = ", Nval, ", I0 = ", I0, ", mu = 1, beta1 = 2.4 / N, beta2 = 4.4 / N^2"}];
plot = Plot[
    {expectedValueKE[t], expectedValueMF[t]}, 
    {t, 0, tMax},
    PlotLegends -> {"Kolmogorov", "Mean-Field"},
    AxesLabel -> {"Time t", "E[X(t)]"},
    PlotLabel -> pltLabel,
    GridLines -> Automatic,
    ImageSize -> Large,
    PlotRange -> {0, 800},
    PlotStyle -> {{Thick, Blue}, {Thick, Dashed, Red}},
    BaseStyle -> {FontSize -> 12}
];

(* Add steady state line *)
steadyState = 750;
plotWithSS = Show[plot, 
    Graphics[{Gray, Dashed, Line[{{0, steadyState}, {tMax, steadyState}}]}]
];

Print["Exporting plot to: ", figurePath];
Export[figurePath, plotWithSS];

(* Verify steady state *)
finalValueKE = expectedValueKE[tMax];
finalValueMF = expectedValueMF[tMax];
Print["E[X(", tMax, ")] from KE: ", finalValueKE];
Print["E[X(", tMax, ")] from MF: ", finalValueMF];

Print["Done."];

(* 
N = 1000, I0 = 50, beta1 = 0.0024, beta2 = 4.4*^-6, mu = 1 
Solving Kolmogorov equations with stiff solver...
Exporting plot to: ../figures/solutions_KEs_MF/.pdf
E[X(10.)] from KE: 752.8512159322886
E[X(10.)] from MF: 753.4091230043464
Done.
*)