

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

; Test the function
