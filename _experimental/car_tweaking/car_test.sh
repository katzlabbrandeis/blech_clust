DIR_LIST=("/media/storage/gc_only/AS18/AS18_4Tastes_200228_151511_old_car/" "/media/storage/gc_only/AS18/AS18_4Tastes_200228_151511_pca_car/")
for DIR in "${DIR_LIST[@]}"; do
    echo $DIR
    python blech_clean_slate.py $DIR
    python blech_clust.py $DIR
    python blech_common_avg_reference.py $DIR
    bash blech_clust_jetstream_parallel.sh
done
