#!/usr/local/bin/wolframscript

(* vary-betas-2.wls *)

ClearAll["Global`*"]
(* :!CodeAnalysis::BeginBlock:: *)
(* :!CodeAnalysis::Disable::SuspiciousSessionSymbol:: *)

figurePath = \
  "../figures/solutions-kolmogorov/test-mathematica/expected-values-varying-betas-2.pdf";

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
(* {odes, vars} = constructODEs[Nval, betaPairs[[1,1]], betaPairs[[1,2]], mu]; *)
(* Print["vars = ", vars, ":\n", odes, "\n"]; *)


(* ----------------------------------------------------- *)
(*  Plot expected values for each (beta1, beta2) pair    *)
(* ----------------------------------------------------- *)
(* put all expected values functions in a list *)
expectedValueFunctionsList = Table[
   Module[{
     beta1 = N[pair[[1]]],
     beta2 = N[pair[[2]]],
     odes,
     vars,
     p0,
     solutionFuncs,
     expectedValueFunc
     },
    Print["Running with current pair (beta1, beta2) = ", beta1, ", ", beta2];
    {odes, vars} = constructODEs[Nval, beta1, beta2, mu];
    p0 = Table[p[k][0] == If[k == I0, 1.0, 0.0], {k, 0, Nval}];
    
    (* Solve the system numerically *)
    (* TODO: adjust MaxSteps *)
    solutionFuncs = NDSolveValue[
                         Join[odes, p0],
                         Table[p[k], {k, 0, Nval}], (* TODO: can't use vars? *)
                         {t, 0, tMax},
                         MaxSteps -> 50000];
    
    
    If[ (* if NDSolve fails to find solution *)
        Head[solutionFuncs] =!= List || Length[solutionFuncs] != Nval + 1,
        Print["NDSolve failed for current beta pair"];
        Missing["NDSolveFailed"],
        (* else, return the expected value, note shift index k -> k + 1 *)
        expectedValueFunc[t_?NumericQ] := Sum[k * solutionFuncs[[k + 1]][t], {k, 0, Nval}];
        Print["Done"];
        expectedValueFunc]
    ], {pair, betaPairs} (* for all (beta1, beta2) pairs *)
];

(* --- Plotting --- *)
figurePath = "../figures/solutions-kolmogorov/test-mathematica/expected-values-varying-betas-2.pdf";

pltLabel = Row[{"Expected values E[X(t)] for varying betas, N=", Nval, ", I0=", I0, ", mu = ", mu}];

(* No legendLabels *)

Print["Plotting..."];
plotExVals = Plot[
   Evaluate[#[t] & /@ expectedValueFunctionsList],
   {t, 0, tMax},
   AxesLabel -> {"Time t", "E[X(t)]"},
   PlotLabel -> pltLabel,
   PlotStyle -> Thick,
   (* PlotLegends -> LineLegend[Automatic, legendLabels], *)
   GridLines -> Automatic,
   ImageSize -> Large,
   PlotRange -> {0, All} (* y-axis should start at 0 *)
];
(* plotExVals *)

Export[figurePath, plotExVals];
Print["Plot saved to: ", figurePath];

(* :!CodeAnalysis::EndBlock:: *)