#!/usr/local/bin/wolframscript

(* Solving forward Kolmogorov equations 

## Listing ODEs approach

  1. define the variables for all states k: 
    
    vars = {p0(t), p1(t), ..., pN(t)}
  
  2. list equations for all states k: 

    odes = {dp0/dt, dp1/dt, ..., dpN/dt}

  3. set initial conditions for p_k(0): 
  
    ics = {p0(0), p1(0), ..., pN(0)}

  4. try to find analytic solution for p_k(t):

    DSolve[Join[odes, ics], vars, t]  
  
  5. else, find numerical solution for p_k(t), 
  but for specific parameter values only:
    beta1 = 2 / N
    beta2 = 4 / (N^2)
    mu = 1

Notes:
- Mathematica fails to find analytic solution
- it only finds the numerical solution

Tested for: 
  - solving: 2.2 Forward Kolmogorov equations - special hypergraphs only
  - complete hypergraph on N = 4 nodes
  - with initial condition: k = 1 infected node at time 0

Results:
  - DSolve didn't find solution
  - NDSove found numerical solution

Saved plots for:
  - probabilities p_{k}(t)
  - expected values E[X(t)]

*)

ClearAll["Global`*"]
(* :!CodeAnalysis::BeginBlock:: *)
(* :!CodeAnalysis::Disable::SuspiciousSessionSymbol:: *)

Nval = 4; (* number of nodes *)

mu = 1; (* fix mu since we don't care about mu. *)

(* rates a_k, b_k, c_k for complete case *)
a[k_] := beta1 k (Nval - k);
b[k_] := beta2 Binomial[k, 2] (Nval - k);
c[k_] := mu k;

vars = Table[p[k][t], {k, 0, Nval}];
Print["vars:\n ", vars, "\n"]

(* list all N + 1 ODE equations *)
odes = Table[
   Which[ (* handle boundry states k = 0 and k = N *)
    k == 0,
    p[k]'[t] == c[k + 1] p[k + 1][t],

    k == Nval,
    p[k]'[t] == (a[k - 1] + b[k - 1]) p[k - 1][t] - c[k] p[k][t],

    True, (* interior states 1 <= k <= N - 1 *)
    p[k]'[t] == (a[k - 1] + b[k - 1]) p[k - 1][t] - (a[k] + b[k] + c[k]) p[k][t] + c[k + 1] p[k + 1][t]
    ],
   {k, 0, Nval} (* for all states k = 0, 1, ..., N *)
];
Print["KEs for N = ", Nval, ":\n", odes, "\n"];

(* Set initial condition, e.g. on infected node k = 1 at time 0 *)
initialState = 1;
ics = Table[p[k][0] == If[k == initialState, 1, 0], {k, 0, Nval}];
Print["Initial condition: \n", ics, "\n"];

(* Try to find analytic solution *)
(* abort and quit, doesn't finish at all:

solution = DSolve[Join[odes, ics], vars, t]
Print["Found analytic solution"];

*)

(* Else, numerical solution only *)
(* Substitute parameter values into ODEs *)
paramValues = {
  beta1 -> 2 / Nval,
  beta2 -> 4 / (Nval^2),
  mu -> 1
};
odesNumerical = odes /. paramValues;
tMax = 10;
solutionNumerical = NDSolve[Join[odesNumerical, ics], vars, {t, 0, tMax}];
Print["Found numerical solution"];

(* plot probabilities p_{k}(t) *)
plotProbsLabel = 
  Row[{"Probability distributions ", 
    TraditionalForm[Subscript[p, k][t]], 
    " for complete hypergraph on N = ", Nval, " nodes"}];
plotProbs = Plot[Evaluate[Table[p[k][t], {k, 0, Nval}] /. solutionNumerical], {t, 0, tMax}, 
  PlotLegends -> Table[TraditionalForm[Subscript[p, k][t]], {k, 0, Nval}],
  AxesLabel -> {"Time t", "Probability"},
  PlotLabel -> plotProbsLabel,
  ImageSize -> Large
]
Export["../figures/solutions-kolmogorov/test-mathematica/probabilities.pdf", plotProbs];

(* extract the list of rules from NDSolve output *)
solRules = solutionNumerical[[1]];
expectedValue[t_] := Sum[k*p[k][t], {k, 0, Nval}] /. solRules;
Print["\nE[ X(0) ] = ", expectedValue[0] /. Rule @@@ ics, "\n"]

(* plot expected values E[X(t)] *)
plotExpValsLabel = 
  Row[{"Expected values E[X(t)], for complete hypergraph on N = ", Nval, " nodes"}];
plotExVals = Plot[expectedValue[t], {t, 0, tMax},
  AxesLabel -> {"Time t", "E[X(t)]"},
  PlotLabel -> plotExpValsLabel,
  PlotStyle -> {Thick, Black},
  GridLines -> Automatic,
  ImageSize -> Large
]
Export["../figures/solutions-kolmogorov/test-mathematica/expected-values.pdf", plotExVals];

(* :!CodeAnalysis::EndBlock:: *)
