#coding:utf-8
# 2018.10.30
# mason_P first nn design for classify

from config import *
from dataset import *
import net_design
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score,



logging.info("**********************mason_p nn_design(%s)***********************"%TM)


def main():
    with tf.device(device),tf.Session() as sess:
        ###input data
        X=tf.placeholder(tf.float32,shape=(None))
        Y=tf.placeholder(tf.float32,shape=(None))
        input=X

        ###set up net /build net
        net=net_design.net.net_build(input)

        ###loss func
        cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,logits=net["output"])
        loss=tf.reduce_mean(cross_entropy,name="loss")
        loss_summary=tf.summary.scalar("loss",loss)


        ###optimization
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss)

        ###acc
        #acc is not for/contain train, it comes from test, test dataset
        predictiton=tf.argmax(net["output"],1) # get the index of the max one
        #Accuracy= tf.nn.in_top_k(net["output"], Y,1)#y is ground_truth,
        correct_prediction=tf.equal(predictiton,tf.argmax(Y,1))
        acc=tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) #reduce_mean get the average of acc; tf.cast change the data type
        acc_summary=tf.summary.scalar("acc",acc)


        ###sess.run/ begin to cal
        #initial variable
        sess.run(tf.global_variables_initializer())
        for epoch in range(epoch_n): # iters= epoch_n * batch_n
            for batch in range(batch_n):
                iters= (batch+1) * (epoch+1)

                if iters % show_iter!=0:
                    #train
                    x_batch,y_batch=random_batch(x_train,y_train,batch_size)
                    train_merged=tf.summary.merge_all()
                    train_summary, _ = sess.run([train_merged,train_op],feed_dict={X: x_batch,Y: y_batch})
                    train_writer = tf.summary.FileWriter(train_logdir)
                    try:
                        train_writer.add_summary(train_summary, iters)
                    except:
                        train_writer.closs()
                        logging.error("add train summary faile, closed writer")


                else:
                    #test, every 100iters
                    test_merged=tf.summary.merge([loss_summary,acc_summary]) #or test_merged= tf.summary.merge_all()
                    test_summary,test_loss, test_acc, test_pred, test_crt_pred,t= sess.run([test_merged,loss, acc, predictiton,correct_prediction],
                                                                              feed_dirt={X: x_test, Y: y_test})
                    # do not need batch, as the testset is much small

                    test_writer = tf.summary.FileWriter(test_logdir)
                    try:
                        test_writer.add_summary(test_summary, global_step=iters)
                    except:
                        test_writer.closs()
                        logging.error("add test summary faile, closed writer")
                    test_precision_score=precision_score(y_test,test_pred)
                    test_recall_score=recall_score(y_test,test_pred)
                    logging.info("epoch:{0}\titers:{1}\ttest_loss:{2}\ttest_acc:{3}\tprecision_score:{4}\trecall_score:{5}"
                                 .format(epoch,iters,test_loss,test_acc,test_precision_score,test_recall_score))

        train_writer.close()
        test_writer.close()






if __name__=="__main__":
    main()