; Fixed SBX Crossover Implementation

; Main SBX crossover function
(= (sbx-crossover $p1 $p2 $eta $crossover_rate)
    (let* (
        ($r (random-float &rng 0 1))  ; Fixed: added &rng
    )
    (if (> $r $crossover_rate)
        ($p1 . $p2)  ; No crossover
        (let* (
            ($child1 (sbx-crossover-loop $p1 $p2 $eta 0 4))
            ($child2 (sbx-crossover-loop $p1 $p2 $eta 0 4))
            ; ($child1 (sbx-crossover-rec $p1 $p2 $eta))
            ; ($child2 (sbx-crossover-rec $p2 $p1 $eta))
        )
        ($child1 . $child2)
        )
    )
))

; Recursive helper to process each gene
; Looping version of SBX crossover using index and pow-math for 1e-14
(= (sbx-crossover-loop $xs $ys $eta $i $n)
    (if (>= $i $n)
        ()
        (let*
            (
                ($x1 (index-atom $xs $i))
                ($x2 (index-atom $ys $i))
                ($diff (abs-math (- $x1 $x2)))
                ($eps (pow-math 10 -14))
                ($tail (sbx-crossover-loop $xs $ys $eta (+ $i 1) $n))
            )
            (if (<= $diff $eps)
                (cons-atom $x1 $tail)
                (let*
                    (
                        ($xlow (min-atom ($x1 $x2)))
                        ($xhigh (max-atom ($x1 $x2)))
                        ($rand (random-float &rng 0 1))
                        ($beta (+ 1.0 (* 2.0 (/ $xlow (- $xhigh $xlow)))))
                        ($beta_power (/ 1.0 (pow-math $beta (+ $eta 1.0))))
                        ($alpha (- 2.0 $beta_power))
                        ($betaq
                            (if (<= $rand (/ 1.0 $alpha))
                                (pow-math (* $rand $alpha) (/ 1.0 (+ $eta 1.0)))
                                (pow-math (/ 1.0 (- 2.0 (* $rand $alpha))) (/ 1.0 (+ $eta 1.0)))))
                        ($child (* 0.5 (- (+ $x1 $x2) (* $betaq (- $x2 $x1)))))
                        ($clamped_child (max-atom (0.0 $child)))
                        ($clipped (min-atom (1.0 $clamped_child)))
                    )
                    (cons-atom $clipped $tail)
                )
            )
        )
    )
)



; Wrapper function
; (= (sbx-crossover-rec $xs $ys $eta)
;     (let* (
;         ($n (size-atom $xs))
;     )
;     (sbx-crossover-loop $xs $ys $eta 0 $n)
;     )
; )

; Test the corrected implementation
!(sbx-crossover (0.7812015273974839 0.9304311460475365 0.8856786021199445 0.9385503728389093) (0.2754761373239567 0.9304311460475365 0.4 0.8385503728389093) 1.05 0.9)