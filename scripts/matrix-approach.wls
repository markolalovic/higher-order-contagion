#!/usr/local/bin/wolframscript

(* Solving forward Kolmogorov equations

## Generator matrix approach

      p'(t) = Q p(t)
      subject to p(0) = p0

  1. define solution: 
  
    p(t) = (p_k (t))_{k} for all states k

  2. construct the generator matrix Q:

    Q = {q_{ij}}, s.t. d/dt p(t) = Q p(t)

  3. set initial condition p(0), e.g. if `i` infected at t = 0:

    p(0) = {0, ... 0, 1 <- `i`th value, 0, ..., 0, 0};
   
  4. try to find analytic solution for p(t) directly as:

    p(t) = Exp[ Q t ] p0

  5. compare it to result using eigendecomposition of Q first:

    Q = U D U^{-1}

    p(t) = U Exp [ D t] U^{-1}

  6. try to simplify p(t) adding assumptions:

    t >= 0, beta1, beta2 >= 0, and beta1, beta2 \in Reals

  7. check the size of expressions for p(t) from (4.) and from (5.), using LeafCount,
  i.e. the number of subexpressions in expression that correspond to leaves on the expression tree:
  
    LeafCount[ p(t) ]
  
Notes:
  - symbolic (analytic) solutions exist since it is a linear ODE system
  - although closed-form expression is huge even for N = 4
 
Tested for: 
  - solving: 2.2 Forward Kolmogorov equations - special hypergraphs only
  - complete hypergraph on N = 4 nodes
  - with initial condition: k = 1 infected node at time 0

Saved plots for:
  - probabilities p_{k}(t)
  - expected values E[X(t)]

*)

ClearAll["Global`*"]
(* :!CodeAnalysis::BeginBlock:: *)
(* :!CodeAnalysis::Disable::SuspiciousSessionSymbol:: *)

pOftFilePath = "/Users/markolalovic/dev/ho_contagion/results/pOftMatrixExpSimplified.mx";

Nval = 4; (* number of nodes *)

mu = 1; (* fix mu since we don't care about mu. *)

(* rates a_k, b_k, c_k for complete case *)
a[k_] := beta1 k (Nval - k);
b[k_] := beta2 Binomial[k, 2] (Nval - k);
c[k_] := mu k;

(* Set initial condition, e . g . one infected node at t = 0 *)
p0 = {0, 1, 0, 0, 0};

(* Construct matrix Q, s . t . d/dt p (t) = Q p (t),  *)
(* where p (t) = (p_k (t)) _ {k} *)
matrixQ = {
   {0, c[1], 0, 0, 0},
   {a[0] + b[0], -(a[1] + b[1] + c[1]), c[2], 0, 0},
   {0, a[1] + b[1], -(a[2] + b[2] + c[2]), c[3], 0},
   {0, 0, a[2] + b[2], -(a[3] + b[3] + c[3]), c[4]},
   {0, 0, 0, a[3] + b[3], -c[4]}
};


(*------------------------------------------*)
(* Symbolically compute p(t) = Exp[ Q t] p0 *)
(*------------------------------------------*)
Print["\nCalculating p(t) using Exp[Q t] p0 ..."];
pOftMatrixExp = MatrixExp[matrixQ t] . p0;
Print["\nFound symbolic solution p(t) = Exp[Q t] p0 "];
Print["LeafCount using MatrixExp: ", LeafCount[pOftMatrixExp]]; (* 34722 ?? *)

(*-------------------------------------------*)
(* Using eigen decomposition U Exp[D t] U^-1 *)
(*-------------------------------------------*)
{eigenvalues, eigenvectors} = Eigensystem[matrixQ];
matrixU = Transpose[eigenvectors];
matrixUinv = Inverse[matrixU]; (* this can give large expression *)
Print["Inverse of U found"];

(* take e^{D \, t} for diagonal matrix `matrixD = [ eigenvals d_{ii} ]_{i} t` *)
expDt = DiagonalMatrix[Exp[eigenvalues t]];

(* check if eigen decomposition is correct *)
Print["Check Q = U D U^-1: ", 
  Simplify[
   matrixQ == matrixU . DiagonalMatrix[eigenvalues] . matrixUinv]
];

Print["\nCalculating p(t) using eigen decomposition U Exp[ D t ] U^-1 p0 ... "];
pOftEigen = matrixU . expDt . matrixUinv . p0;
Print["\nFound symbolic solution p(t) = U Exp[ D t ] U^-1 p0 "]; 
Print["LeafCount using eigendecomposition: ", LeafCount[pOftEigen]]; (* 430767 ?? *)

(*-------------------------------------------*)
(* Try to simplify solutions found           *)
(*-------------------------------------------*)
(* Takes a while: 
Print["Trying to simplify solutions found ..."] 
(* Define assumptions to simplify *)
assumptions = {t >= 0, beta1 >= 0, beta2 >= 0, Element[{beta1, beta2}, Reals]}

(* Simplify the results *)
pOftMatrixExpSimplified = Simplify[pOftMatrixExp, Assumptions -> assumptions];
pOftEigenSimplified = Simplify[pOftEigen, Assumptions -> assumptions];

(* Compare leaf counts *)
Print["LeafCount of p(t) matrix exp directly and simplified: ", LeafCount[pOftMatrixExpSimplified]];
Print["LeafCount of p(t) eigen decomposed and simplified: ", LeafCount[pOftEigenSimplified]];
(* LeafCount of p(t) matrix exp directly and simplified: 22454 *)
(* LeafCount of p(t) eigen decomposed and simplified: 102850 *)

(* Save the simplified p(t) *)
Export[pOftFilePath, pOftMatrixExpSimplified, "MX"]
*)

(*-------------------------------------------*)
(* Plot the solution                         *)
(*-------------------------------------------*)
(* Load the simplified p(t) *)
pOft = Import[pOftFilePath];
Print["LeafCount of p(t) matrix exp directly and simplified: ", LeafCount[pOft]];

(* Set parameter values for plots *)
paramValues = {
   beta1 -> 2 / Nval,
   beta2 -> 4 / (Nval^2)
};
pOftAtParamValues = pOft /. paramValues;

(* And set time tMax! *)
tMax = 10;

(* Plot probabilities p_{k}(t) *)
plotProbsLabel = 
  Row[{"Evaluated symbolic solution probability distributions ", 
    TraditionalForm[Subscript[p, k][t]], 
    " for complete hypergraph on N = ", Nval, " nodes"}];
plotProbs = Plot[Evaluate[pOftAtParamValues], {t, 0, tMax},
  PlotLegends -> Table[TraditionalForm[Subscript[p, k][t]], {k, 0, Nval}],
  AxesLabel -> {"Time t", "Probability"},
  PlotLabel -> plotProbsLabel,
  ImageSize -> Large
];
Export["../figures/solutions-kolmogorov/test-mathematica/probabilities-symbolic-eval.pdf", plotProbs];


(* extract expected values *)
(* Note: ! indexes go from 1 ! *)
expectedValue[t_] := Sum[k * pOftAtParamValues[[k + 1]], {k, 0, Nval}];

(* plot expected values E[X(t)] *)
plotExpValsLabel = 
  Row[{"Evaluated symbolic solution expected values E[X(t)], for complete hypergraph on N = ", Nval, " nodes"}];
plotExVals = Plot[expectedValue[t], {t, 0, tMax},
  AxesLabel -> {"Time t", "E[X(t)]"},
  PlotLabel -> plotExpValsLabel,
  PlotStyle -> {Thick, Black},
  GridLines -> Automatic,
  ImageSize -> Large
];
Export["../figures/solutions-kolmogorov/test-mathematica/expected-values-symbolic-eval.pdf", 
  plotExVals]; 

(* :!CodeAnalysis::EndBlock:: *)
