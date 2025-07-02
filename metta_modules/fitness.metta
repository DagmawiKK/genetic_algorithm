(= (fitness-rec $cand $a $b $i $gene_len)
    (if (>= $i $gene_len)
        0
        (let*
            (
                ($c (car $cand))
                ($a1 (car $a))
                ($b1 (car $b))
                ($maxAB (max2 $a1 $b1))
                ($diff (sub $c $maxAB))
                ($emergence (max0 $diff))
                ($minAB (min2 $a1 $b1))
                ($contrib (mul $minAB $emergence))
                ($rest (fitness-rec (cdr $cand) (cdr $a) (cdr $b) (+ $i 1) $gene_len))
            )
            (add $contrib $rest)
        )
    )
)

(= (fitness $individual $input_a $input_b $gene_len)
    (let*
        (
            ($total (fitness-rec $individual $input_a $input_b 0 $gene_len))
            ($normalized (div $total $gene_len))
            ($clamped (min2 $normalized 1.0))
        )
        $clamped
    )
)
