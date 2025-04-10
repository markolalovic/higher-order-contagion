#!/usr/local/bin/wolframscript

(* vary-betas.wls *)

ClearAll["Global`*"]
(* :!CodeAnalysis::BeginBlock:: *)
(* :!CodeAnalysis::Disable::SuspiciousSessionSymbol:: *)

figurePath = \
  "../figures/solutions-kolmogorov/test-mathematica/expected-values-varying-betas.pdf";

(* --- Setup --- *)
mu = 1.0;    (* fixed recovery rate *)
tMax = 20;   (* maximum time for plots - increase, needs to be long enough *)

Nval = 100;
I0 = 10;
Print["N = ", Nval, ", I0 = ", I0, ", mu = ", mu];

(* check for various values of (beta1, beta2) pairs *)
betaPairs = {
  {0.023, 0.00042},
  {2.0 / Nval, 4.0 / Nval^2},
  {2.0 / Nval, 3.0 / Nval^2},
  {2.0 / Nval, 2.0 / Nval^2},    
  {1.5 / Nval, 3.0 / Nval^2},
  {0.01, 0.0005}  
}

(* ----------------------------------------------------- *)
(*  Construct generator matrix Q                         *)
(* ----------------------------------------------------- *)
(* Takes: N, beta1, beta2, mu, as arguments *)
constructQ[n_, beta1_, beta2_, mu_] := Module[{matrixQ, k},
  (* define rates a_k, b_k, c_k for given n *)
  Clear[a, b, c]; (* clear any previous definitions *)
  a[k_] := beta1 k (n - k);
  b[k_] := beta2 Binomial[k, 2] (n - k);
  c[k_] := mu k;
  
  (* TODO: initialize sparse matrix of size (n + 1) x (n + 1) for performance *)
  (* Q = SparseArray[{}, {n + 1, n + 1}]; *)
  matrixQ = Table[
    Which[
     k == 0 && j == 1, c[1],
     k == j + 1, a[j] + b[j],
     k == j, -(a[k] + b[k] + c[k]),
     k + 1 == j, c[j], True, 0
     ], {k, 0, n}, {j, 0, n}
    ];
  matrixQ];


(* ----------------------------------------------------- *)
(*  Plot expected values for each (beta1, beta2) pair    *)
(* ----------------------------------------------------- *)

(* put all expected values functions in a list *)
expectedValueFunctionsList = Table[
   Module[{
    currentBeta1 = pair[[1]], 
    currentBeta2 = pair[[2]], 
    currentQ, 
    pOftFunc, 
    expectedValueFunc},
    
    Print["\nRunning with beta1 = ", currentBeta1, " and beta2 = ", currentBeta2];
    currentQ = constructQ[Nval, currentBeta1, currentBeta2, mu];
    currentP0 = Table[If[k == I0, 1, 0], {k, 0, Nval}];
    
    pOftFunc[t_] = MatrixExp[currentQ t] . currentP0;
    
    expectedValueFunc[t_] := Sum[k * pOftFunc[t][[k + 1]], {k, 0, Nval}];
    Print["Done.\n"];
    expectedValueFunc
    ],
    {pair, betaPairs} (* iterate through (beta1, beta2) pairs *)
  ];

pltLabel = 
  Row[{"Expected values E[X(t)] for varying beta1, beta2, and I0 = ", I0, ", N = ", Nval}];


legendLabels = Table[
  "beta1, beta2 " <> ToString[betaPairs[[i]]],
  {i, Length[betaPairs]}
];

Print["\nPlotting ..."];
plotExVals = Plot[
  Evaluate[#[t] & /@ expectedValueFunctionsList],
  {t, 0, tMax},
  AxesLabel -> {"Time t", "E[X(t)]"},
  PlotLabel -> pltLabel,
  PlotStyle -> Thick,
  PlotLegends -> LineLegend[Automatic, legendLabels],
  (* PlotLegends -> Placed[LineLegend[legendLabels], {1.1, 0.8}],   *)
  GridLines -> Automatic,
  ImageSize -> Large,
  PlotRange -> {0, All} (* y-axis should start at 0 *)
  ];

Export[figurePath, plotExVals];
Print["Done, saved to: ", figurePath];

(* :!CodeAnalysis::EndBlock:: *)  