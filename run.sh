#-------------------------------------------------------------------------------
# Run StackNet
#-------------------------------------------------------------------------------

echo "Running StackNet.jar..."
java -Xmx3048m -jar StackNet.jar train train_file=data/train_std.csv test_file=data/test_std.csv params=params.txt pred_file=top_pred.csv output_name=stacknet_oof test_target=true verbose=true Threads=3 stackdata=false folds=5 seed=1 metric=logloss
