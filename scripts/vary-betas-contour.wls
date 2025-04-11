#!/usr/local/bin/wolframscript

(* vary-betas-contour.wls *)
(* --- Contour plot --- *)

ClearAll["Global`*"]
(* :!CodeAnalysis::BeginBlock:: *)
(* :!CodeAnalysis::Disable::SuspiciousSessionSymbol:: *)

figurePath = "../figures/solutions-kolmogorov/test-mathematica/expected-values-varying-betas-contour.pdf";

(* --- Setup --- *)
Nval = 100; (* number of nodes *)
I0 = 10; (* initial infected *)

tMax = 20; (* TODO: could increase to 100 *)
mu = 1; (* recovery rate mu is fixed *)
Print["N = ", Nval, ", I0 = ", I0, ", mu = ", mu, ", tMax = ", tMax];

(* Check for various values of (beta1, beta2) pairs *)
iMax = 9;
jMax = 9;

(* For testing *)
(* iMax = 1; *)
(* jMax = 2; *)

eps = 10^-1; (* shift for esp to not start with 0 *)
beta1Vec = Table[N[(i + eps) / Nval], {i, 0, iMax}];
beta2Vec = Table[N[(j + eps) / (Nval^2)], {j, 0, jMax}];

(* Create all combinations (grid points) using `Tuples` *)
betaPairs = Tuples[{beta1Vec, beta2Vec}];

Print["Generated ", Length[betaPairs], " (beta1, beta2) pairs: "];
Print["beta1 \t beta2"]
Print[MatrixForm[Take[betaPairs, 5]]];


(* ----------------------------------------------------- *)
(*  Construct a list of ODEs                             *)
(* ----------------------------------------------------- *)
constructODEs[n_, beta1_, beta2_, mu_] := Module[{odes, vars, k},
  Clear[a, b, c, p];
  a[k_] := beta1 k (n - k);
  b[k_] := beta2 Binomial[k, 2] (n - k);
  c[k_] := mu k;

  vars = Table[p[k][t], {k, 0, n}];

  odes = Table[
    Which[
     (* Boundary state k = 0 *)
     k == 0,
     p[k]'[t] == c[k + 1] p[k + 1][t],

     (* Boundary state k = N *)
     k == n,
     p[k]'[t] == (a[k - 1] + b[k - 1]) p[k - 1][t] - c[k] p[k][t],

     (* Interior states 1 <= k <= N - 1 *)
     True,
     p[k]'[t] == (a[k - 1] + b[k - 1]) p[k - 1][t] - (a[k] + b[k] + c[k]) p[k][t] + c[k + 1] p[k + 1][t]
     ],
    {k, 0, n}
    ];

  {odes, vars} (* Return both odes and vars *)
];

(* Evaluate each expected value function at tMax *)
(* Calculate expected values at tMax for each beta1, beta2 pair *)
expectedValuesAtTMaxList = Table[
   Module[{
     beta1 = N[pair[[1]]],
     beta2 = N[pair[[2]]],
     odes,
     vars,
     p0,
     solutionFuncs,
     expectedValueAtTMax (* Changed variable name *)
     },
    Print["Running for (beta1, beta2) = ", {beta1, beta2}];
    
    (* Construct ODEs and Initial Conditions *)
    {odes, vars} = constructODEs[Nval, beta1, beta2, mu];
    p0 = Table[p[k][0] == If[k == I0, 1.0, 0.0], {k, 0, Nval}];
    
    (* Solve the system numerically *)
    (* TODO: adjust MaxSteps *)
    solutionFuncs = NDSolveValue[
                          Join[odes, p0],
                          Table[p[k][t], {k, 0, Nval}], (* Get expressions p[k][t] *)
                          {t, 0, tMax},
                          MaxSteps -> 50000];
    
    If[ (* if NDSolve fails to find solution *)
        Head[solutionFuncs] =!= List || Length[solutionFuncs] != Nval + 1,
        Print["NDSolve failed for current beta pair"];
        Missing["NDSolveFailed"],
        (* else, return the expected value, note shift index k -> k + 1 *)
        expectedValueAtTMax = Sum[k * (solutionFuncs[[k + 1]] /. t -> tMax), {k, 0, Nval}];
        Print["Done"];
        expectedValueAtTMax]
   ], {pair, betaPairs} (* for all (beta1, beta2) pairs *)
];

contourPlotData = MapThread[
   { N[#1[[1]]], N[#1[[2]]], #2 } &, 
   {betaPairs, expectedValuesAtTMaxList}
];

Print["Plotting ..."];
pltLabel = Row[{"Stationary state k* = E[X(tMax)], where: tMax = ", tMax, ", N=", Nval, ", I0=", I0, ", mu = ", mu}];
contourPlot = ListContourPlot[
    contourPlotData,
    ContourLabels -> True,
    PlotLegends -> Automatic,
    FrameLabel -> {"beta1","beta2"},
    PlotLabel -> pltLabel,
    ColorFunction -> "TemperatureMap",
    ImageSize -> Large,
    AspectRatio -> 1
];
(* contourPlot *)

Export[figurePath, contourPlot];
Print["Plot saved to: ", figurePath];

(* :!CodeAnalysis::EndBlock:: *)
