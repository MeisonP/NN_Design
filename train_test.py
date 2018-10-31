#coding:utf-8
# 2018.10.30
# mason_P first nn design for classify

from config import *
from dataset import *
import net_design
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score
import os



def main():
    with tf.device(device),tf.Session() as sess:
        ###input data / placeholder
        X=tf.placeholder(tf.float32,shape=(None,x_train.shape[1]))
        Y=tf.placeholder(tf.int32)

        ###set up net /build net
        net=net_design.network.net_build(input_=X, output_dim=classes)

        ###loss func
        cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,logits=net["output"])
        loss=tf.reduce_mean(cross_entropy,name="loss")
        loss_summary=tf.summary.scalar("loss",loss)


        ###optimization
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss)

        ###acc
        #acc is not for/contain train, it comes from test, test dataset
        prediction=tf.argmax(net["output"],1) # get the index of the max one
        correct_prediction= tf.nn.in_top_k(net["output"], Y,1)#y is ground_truth,
        #correct_prediction=tf.equal(prediction,tf.argmax(Y,1))
        acc=tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) #reduce_mean get the average of acc; tf.cast change the data type
        acc_summary=tf.summary.scalar("acc",acc)


        ###sess.run/ begin to cal
        #checkpoint
        saver = tf.train.Saver(max_to_keep=5) #keep last 4
        #summary writer
        train_writer = tf.summary.FileWriter(train_logdir)
        test_writer = tf.summary.FileWriter(test_logdir)

        #initial variable,
        if os.path.isfile(iter_counter_path):
            with open(iter_counter_path, "rb") as f:
                iter_i = int(f.read())
                epoch_start = int(np.ceil(iter_i / batch_n))
                batch_start= iter_i%batch_n
            logging.info("Training was interrupted. Continuing at iter: {}".format(iter_i))
            saver.restore(sess, checkpoint_save_path)
        else:
            epoch_start=0
            batch_start=0
            sess.run(tf.global_variables_initializer())

        for epoch_i in range(epoch_start,epoch_n): # start from 0 to (epoch_n -1)
            for batch_i in range(batch_start,batch_n):
                iter= batch_n* epoch_i + batch_i # 1 iter is one batch pass through cal;iter starts from 1, not 0
                # iter counter
                with open(iter_counter_path, "wb") as f:
                    f.write(b"%d" % iter)  # b mean binary

                if iter % show_iter!=0:
                    #train
                    x_batch,y_batch=random_batch(x_train,y_train,batch_size)
                    train_merged=tf.summary.merge_all()
                    train_summary, _ = sess.run([train_merged,train_op],feed_dict={X: x_batch,Y: y_batch})

                    try:
                        train_writer.add_summary(train_summary, iter)
                    except:
                        train_writer.closs()
                        logging.error("add train summary failed, closed writer")


                else:
                    #test, every 100iter
                    test_merged=tf.summary.merge([loss_summary,acc_summary]) #or test_merged= tf.summary.merge_all()
                    test_summary,test_loss, test_acc, test_pred, test_crt_pred= sess.run([test_merged,loss, acc, prediction,correct_prediction],
                                                                              feed_dict={X: x_test, Y: y_test})
                    # do not need batch, as the testset is much small

                    try:
                        test_writer.add_summary(test_summary, global_step=iter)
                    except:
                        test_writer.closs()
                        logging.error("add test summary failed, closed writer")

                    #logging.info("{}\n{}\n{}".format(y_test, test_pred,test_crt_pred))
                    test_precision_score=precision_score(y_test,test_crt_pred)
                    test_recall_score=recall_score(y_test,test_crt_pred)
                    logging.info("epoch:{0}\titer:{1}\ttest_loss:{2}\ttest_acc:{3}\tprecision_score:{4}\trecall_score:{5}"
                                 .format(epoch_i,iter,test_loss,test_acc,test_precision_score,test_recall_score))

                    #save iter model
                    if iter%checkpoint_iter==0 and iter/checkpoint_iter >=0:
                        saver.save(sess,checkpoint_save_path,global_step=iter)#the global_step tell which model to save






        train_writer.close()
        test_writer.close()

        ###save final model; there can be some mechanism on here model save
        saver.save(sess,final_model_path)







if __name__=="__main__":
    logging.info("**********************mason_p nn_design(%s)***********************" % TM)
    main()