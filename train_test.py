# conding: utf-8
# 2018.10.30
# mason_P first nn design for classify,hahah

from config import *
from dataset import *
import net_design
import tensorflow as tf



logging.info("**********************mason_p nn_design(%s)***********************"%TM)


def main():
    with tf.device(device),tf.Session() as sess:
        ###input data
        X=tf.placehoder(tf.float32,shape=(None))
        Y=tf.placehoder(tf.float32,shape=(None))
        input=X

        ###set up net /build net
        net=net_design.net.net_build(input)

        ###loss func
        xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,logits=net["output"])
        loss=tf.reduce_mean(xentropy,name="loss")
        loss_summary=tf.summary.scalar("loss",loss)


        ###optimization
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss)

        ###acc
        #acc is not for/contain train, it comes from test, test dataset
        predict=tf.argmax(net["output"],1) # get the index of the max one
        Accuracy= tf.nn.in_top_k(net["output"], Y,1)#y is ground_truth,
        acc=tf.reduce_mean(tf.cast(Accuracy,tf.float32)) #reduce_mean get the average of acc; tf.cast change the data type
        acc_summary=tf.summary.scalar("acc",acc)


        ###sess.run/ begin to cal
        #initial variable
        sess.run(tf.global_variables_initializer())
        for epoch in range(epoch_n): # iters= epoch_n * batch_n
            for batch in range(batch_n):
                iters= (batch+1) * (epoch+1)
                #train
                x_batch,y_batch=random_batch(x_train,y_train,batch_size)
                sess.run(train_op,feed_dict={X: x_batch,Y: y_batch})
                train_loss=sess.run(loss)


                #test
                test_loss,test_acc,pred=sess.run([loss, acc,predict],feed_dirt={X:x_test, Y:y_test})#do not need batch, as the testset is much small

                #record and summary of train
                summary_op = tf.summary.merge([loss_summary, acc_summary])  # or "tf.summary.merge_all()"
                summary_writer = tf.summary.FileWriter(logdir)
                summary_train=sess.run(summary_op, feed_dict={X: x_batch, Y:y_batch} )
                summary_test = sess.run(summary_op, feed_dict={X: x_test, Y:y_test} )
                try:
                    summary_writer.add_summary([summary_test,summary_train], global_step=None)
                except:
                    summary_writer.closs()
                    logging.error("add summary faile, closed writer")

                if iters%show_iter ==0:
                    logging.info("epoch:{0}\titers:{1}\ttrain_loss:{2}\ttest_loss:{3}\tacc:{4}".format(epoch,iters,train_loss,test_loss,test_acc))
        summary_writer.close()






if __name__=="__main__":
    main()