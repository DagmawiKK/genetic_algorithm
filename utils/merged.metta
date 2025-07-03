!(import! &self rand)
!(bind! &POP_SIZE 4)
!(bind! &GENES 4)
!(bind! &GENERATIONS 5)
!(bind! &ELITE_COUNT 2)
!(bind! &CROSSOVER_RATE 0.9)
!(bind! &MUTATION_RATE 0.1)
!(bind! &INITIAL_MUTATION_STD 0.5)
!(bind! &MUTATION_DECAY 0.95)
!(bind! &INITIAL_SBX_ETA 2)
!(bind! &SBX_ETA_GROWTH 1.05)
!(bind! &input_a (0.9 0.9 0.0 0.8))
!(bind! &input_b (0.8 0.2 1.0 0.3))

; helper functions
(= (car $x) (car-atom $x))
(= (cdr $x) (cdr-atom $x))
(= (cons $x $xs) (cons-atom $x $xs))

; quick-sort
(= (tuple-first-less $a $b)
   (< (car-atom $a) (car-atom $b)))

(= (tuple-first-gte $a $b)
   (>= (car-atom $a) (car-atom $b)))

(= (filter-less $xs $pivot)
   (if (== $xs ()) ()
       (let* (( $x (car $xs) ) ($rest (cdr $xs)))
       (if (tuple-first-less $x $pivot)
           (cons $x (filter-less $rest $pivot))
           (filter-less $rest $pivot)))))

(= (filter-gte $xs $pivot)
   (if (== $xs ()) ()
       (let* (( $x (car $xs) ) ($rest (cdr $xs)))
       (if (tuple-first-gte $x $pivot)
           (cons $x (filter-gte $rest $pivot))
           (filter-gte $rest $pivot)))))

(= (append $xs $ys)
   (if (== $xs ()) $ys
       (cons (car $xs) (append (cdr $xs) $ys))))

(= (quick-sort $xs)
   (if (== $xs ()) ()
       (let* (
           ($pivot (car $xs))
           ($rest (cdr $xs))
           ($less (quick-sort (filter-less $rest $pivot)))
           ($greater (quick-sort (filter-gte $rest $pivot)))
       )
       (append $greater (cons $pivot $less)))))

; max 
(= (max-by-fst $pairs)
   (if (== $pairs ()) ()
       (max-by-fst-rec (cdr $pairs) (car $pairs))))

(= (max-by-fst-rec $pairs $best)
   (if (== $pairs ())
       $best
       (let* (
           ($head (car $pairs))
           ($tail (cdr $pairs))
           ($best (if (> (car-atom $head) (car-atom $best)) $head $best))
       )
       (max-by-fst-rec $tail $best))))
; zip
(= (zip $xs $ys)
   (if (or (== $xs ()) (== $ys ()))
       ()
       (cons (cons (car $xs) (cons (car $ys) ()))  ; Create pair (x y)
             (zip (cdr $xs) (cdr $ys)))))

; slice
(= (slice $xs $n)
   (if (or (== $xs ()) (<= $n 0))
       ()
       (cons (car $xs) (slice (cdr $xs) (- $n 1)))))

; population initializer
(= (randfloat-array $gene_len)
   (if (<= $gene_len 0)
       ()
       (cons (random-float &rng 0 1) 
             (randfloat-array (- $gene_len 1)))))


(= (make-pop-rec $i $pop_size $gene_len)
   (if (>= $i $pop_size)
       ()
       (cons (randfloat-array $gene_len) 
             (make-pop-rec (+ $i 1) $pop_size $gene_len))))

(= (make-pop $pop_size $gene_len)
   (make-pop-rec 0 $pop_size $gene_len))

; fitness
(= (fitness-rec $cand $a $b $i $gene_len)
    (if (>= $i $gene_len)
        0
        (let*
            (
                ($c (car $cand))
                ($a1 (car $a))
                ($b1 (car $b))
                ($maxAB (max-atom ($a1 $b1)))
                ($diff (- $c $maxAB))
                ($emergence (max-atom ($diff 0)))
                ($minAB (min-atom ($a1 $b1)))
                ($contrib (* $minAB $emergence))
                ($rest (fitness-rec (cdr $cand) (cdr $a) (cdr $b) (+ $i 1) $gene_len))
            )
            (+ $contrib $rest)
        )
    )
)

(= (fitness $individual $input_a $input_b $gene_len)
    (let*
        (
            ($total (fitness-rec $individual $input_a $input_b 0 $gene_len))
            ($normalized (/ $total $gene_len))
            ($clamped (min-atom ($normalized 1.0)))
        )
        $clamped
    )
)

; roulette wheel selection
(= (roulette-stochastic-acceptance $population $fitnesses $len)
   (let* 
       (
           ($w_max (max-atom $fitnesses))
       )
       (if (<= $w_max 0)
           (index-atom $population (random-int &rng 0 (- $len 1)))
           (roulette-loop $population $fitnesses $w_max $len)
       )
   ))

(= (roulette-loop $population $fitnesses $w_max $len)
   (let* 
       (
           ($i (random-int &rng 0 (- $len 1)))
           ($prob (random-float &rng 0 1))
           ($fitness_i (index-atom $fitnesses $i))
           ($ratio (/ $fitness_i $w_max))
       )
       (if (<= $prob $ratio)
           (index-atom $population $i)
           (roulette-loop $population $fitnesses $w_max $len)
       )
   )
)

; sbx crossover
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

; mutation
(= (mutate-rec $ind $std $mutation-rate $i $len)
   (if (>= $i $len)
       ()
       (let* (
           ($gene (index-atom $ind $i))
           ($r (random-float &rng 0 1))
           ($mutated
               (if (< $r $mutation-rate)
                   (let* (
                       ($noise (gaussian-random 0 $std))
                       ($added (+ $gene $noise))
                       ($clipped (max-atom (0 (min-atom (1 $added)))))
                   )
                       $clipped
                   )
                   $gene
               )
           )
           ($rest (mutate-rec $ind $std $mutation-rate (+ $i 1) $len))
       )
           (cons-atom $mutated $rest)
       )
   )
)

(= (mutate $ind $std $mutation-rate)
    (mutate-rec $ind $std $mutation-rate 0 (size-atom $ind)))

; genetic algorithm
(= (genetic-algorithm $input_a $input_b)
   (let* (
       ($population (make-pop &POP_SIZE &GENES))
       ($mutation_std &INITIAL_MUTATION_STD)
       ($sbx_eta &INITIAL_SBX_ETA)
   )
   (genetic-loop 0 $population $mutation_std $sbx_eta $input_a $input_b)))

(= (genetic-loop $gen $population $mutation_std $sbx_eta $input_a $input_b)
    (if (>= $gen &GENERATIONS)
        (let* (
            ($fitnesses (map-atom $population $ind (fitness $ind $input_a $input_b &GENES)))
            ($best_pair (max-by-fst (zip $fitnesses $population)))
            ($best_ind (cdr $best_pair))
        ) $best_ind)
       (let* (
           ($fitnesses (map-atom $population $ind (fitness $ind $input_a $input_b &GENES)))
           ($zipped (zip $fitnesses $population))
           ($sorted (quick-sort $zipped))
           ($elites (slice $sorted &ELITE_COUNT))
           ($elite_inds (map-atom $elites $p (cdr $p)))
           ($newpop (fill-next-gen $elite_inds $population $fitnesses $mutation_std $sbx_eta))
           ($new_eta (* $sbx_eta &SBX_ETA_GROWTH))
           ($new_mut_std (* $mutation_std &MUTATION_DECAY))
       )
       (genetic-loop (+ $gen 1) $newpop $new_mut_std $new_eta $input_a $input_b))))

(= (fill-next-gen $newpop $pop $fitnesses $mutation_std $sbx_eta)
   (if (>= (size-atom $newpop) &POP_SIZE)
       (slice $newpop &POP_SIZE)
       (let* (
           ($p1 (roulette-stochastic-acceptance $pop $fitnesses (size-atom $pop)))  
           ($p2 (roulette-stochastic-acceptance $pop $fitnesses (size-atom $pop)))  
           (($c1 . $c2) (sbx-crossover $p1 $p2 $mutation_std &CROSSOVER_RATE))  
           ($m1 (mutate $c1 $mutation_std &MUTATION_RATE))
           ($m2 (mutate $c2 $mutation_std &MUTATION_RATE))
           ($next (append $newpop (cons $m1 (cons $m2 ()))))
       )
       (fill-next-gen $next $pop $fitnesses $mutation_std $sbx_eta))))

!(genetic-algorithm &input_a &input_b)