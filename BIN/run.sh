#        1_NN
./main ecoli -1nn
./main parkinsons -1nn
./main breast-cancer -1nn

#        GREEDY
./main ecoli -gr
./main parkinsons -gr
./main breast-cancer -gr

#        BL
./main ecoli -bl 42
./main parkinsons -bl 42
./main breast-cancer -bl 42

#        AGG
./main ecoli -agg 42
./main parkinsons -agg 42
./main breast-cancer -agg 42

#        AGG_BLX
./main ecoli -agg_blx 42
./main parkinsons -agg_blx 42
./main breast-cancer -agg_blx 42

#        AGE
./main ecoli -age 42
./main parkinsons -age 42
./main breast-cancer -age 42

#        AGE_BLX
./main ecoli -age_blx 42
./main parkinsons -age_blx 42
./main breast-cancer -age_blx 42

