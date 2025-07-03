(= (sbx-crossover $p1 $p2 $eta $crossover_rate)
    (let* (
        ($r (random-float &rng 0 1))
    )
    (if (> $r $crossover_rate)
        ($p1 . $p2)
        (let* (
            ($child1 (sbx-crossover-loop $p1 $p2 $eta 0 (size-atom $p1) 1))
            ($child2 (sbx-crossover-loop $p1 $p2 $eta 0 (size-atom $p1) 2))
        )
        ($child1 . $child2)
        )
    )
))

(= (sbx-crossover-loop $xs $ys $eta $i $n $child_id)
    (if (>= $i $n)
        ()
        (let*
            (
                ($x1 (index-atom $xs $i))
                ($x2 (index-atom $ys $i))
                ($diff (abs-math (- $x1 $x2)))
                ($eps (pow-math 10 -14))
                ($tail (sbx-crossover-loop $xs $ys $eta (+ $i 1) $n $child_id))
            )
            (if (<= $diff $eps)
                (let* (
                    ($val (if (== $child_id 1) $x1 $x2)) 
                )
                    (cons-atom $val $tail)
                )
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
                        ($c_val
                            (if (== $child_id 1)
                                (* 0.5 (- (+ $x1 $x2) (* $betaq (- $x2 $x1))))
                                (* 0.5 (+ (+ $x1 $x2) (* $betaq (- $x2 $x1)))))
                        )
                        ($clamped_child (max-atom (0.0 $c_val)))
                        ($clipped (min-atom (1.0 $clamped_child)))
                    )
                    (cons-atom $clipped $tail)
                )
            )
        )
    )
)

