#!/usr/local/bin/wolframscript

(* vary-N-size-1.wls

Testing with small values of N: {4, 5, 6}

*)

ClearAll["Global`*"]
(* :!CodeAnalysis::BeginBlock:: *)
(* :!CodeAnalysis::Disable::SuspiciousSessionSymbol:: *)

figurePath = \
  "../figures/solutions-kolmogorov/test-mathematica/expected-values-varying-N-1.pdf";

(* --- Setup --- *)
mu = 1;          (* fixed recovery rate *)
tMax = 10;       (* maximum time for plots *)

(* --- Fixed parameters --- *)
I0 = 1;          (* fixed initial number of infected nodes *)

(* --- Parameter(s) to vary --- *)
(* reversed, so the labels match curves on the figure, top curve = top label *)
Nvalues = Reverse[{4, 5, 6}]; (* number of nodes in H *)

(* ----------------------------------------------------- *)
(*  Function to construct generator matrix Q             *)
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
(*--- constructQ module test ---*)
(* Test with i = 1 
Print["\nTest with i = 1: "];
i = 1;
currentNval = Nvalues[[i]];
currentBeta1 = 2.0 / currentNval;
currentBeta2 = 4.0 / (currentNval^2);

currentQ = constructQ[currentNval, currentBeta1, currentBeta2, mu];
currentP0 = Table[If[k == 1, 1, 0], {k, 0, currentNval}];

pOftFunc[t_] = MatrixExp[currentQ t] . currentP0;

expectedValueN[t_] := Sum[k*pOftFunc[t][[k + 1]], {k, 0, currentNval}];

plotExpValsLabel = Row[
  {"Expected values E[X(t)] for varying N, I0 = ", I0, 
  " where mu = 1, beta1 = 2/N, beta2 = 4/N^2"}];

plotExVals = Plot[expectedValueN[t], {t, 0, tMax},
 AxesLabel -> {"Time t", "E[X(t)]"},
 PlotLabel -> plotExpValsLabel,
 PlotStyle -> Thick,
 GridLines -> Automatic,
 ImageSize -> Large
]
Export["../figures/solutions-kolmogorov/test-mathematica/expected-values-constructQ-module-test.pdf", 
  plotExVals]; 
*)


(* ----------------------------------------------------- *)
(*  Plot expected values for each Nval in Nvalues        *)
(* ----------------------------------------------------- *)

(* put all expected values functions in a list *)
expectedValueFunctionsList = Table[
   Module[{currentNval, currentBeta1, currentBeta2, currentQ, 
     pOftFunc, expectedValueN},
    
    currentNval = Nvalues[[i]];
    currentBeta1 = 2.0 / currentNval;
    currentBeta2 = 4.0 / (currentNval^2);
    
    currentQ = constructQ[currentNval, currentBeta1, currentBeta2, mu];
    currentP0 = Table[If[k == I0, 1, 0], {k, 0, currentNval}];
    
    pOftFunc[t_] = MatrixExp[currentQ t] . currentP0;
    
    expectedValueN[t_] := 
     Sum[k * pOftFunc[t][[k + 1]], {k, 0, currentNval}];
    
    expectedValueN], 
   {i, Length[Nvalues]}];

(* legendLabels *)
legendLabels = ("N = " <> ToString[#]) & /@ Nvalues;
(* legendLabels *)

pltLabel = 
  Row[{"Expected values E[X(t)] for varying N, and fixed: 
  I0 = ", I0, ", mu = 1, beta1 = 2 / N, beta2 = 4 / N^2"}];

plotExVals = Plot[Evaluate[Table[expectedValueFunctionsList[[i]][t], {i, Length[Nvalues]}]], {t, 0, tMax},
  AxesLabel -> {"Time t", "E[X(t)]"},
  PlotLabel -> pltLabel,
  PlotStyle -> Thick,
  PlotLegends -> Placed[LineLegend[legendLabels], {0.8, 0.75}],
  GridLines -> Automatic, 
  ImageSize -> Large];

Export[figurePath, plotExVals];

Print["Done, saved to: ", figurePath];
(* :!CodeAnalysis::EndBlock:: *)  