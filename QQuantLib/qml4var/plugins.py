"""
Plugins for Quantum Machine Learning
using myqlm
"""

import copy
import numpy as np
from qat.plugins import AbstractPlugin, Junction
from qat.core import BatchResult, Result
from qat.core.qpu import QPUHandler

class SetParametersPlugin(AbstractPlugin):
    """
    Pluging for setting the weights and features of the different PQCs.
    The Batch MUST have a meta_data with at least the keys:
    fetures and weights

    Parameters
    ----------

    weights : numpy array
        Array with the weights of the model
    features : numpy array
        Array with the features for the model
    """

    def __init__(self, weights, features):
        """
        Init method
        """
        self.weights = weights
        self.features = features
        #print(self.weights)

    def compile(self, batch, hardware_specs):
        """
        Loop over all the jobs of the input Batch and overwrite
        the different weights of the PQC

        Parameters
        ----------

        batch : QLM Batch
            QLM Batch object with the jobs to execute

        Returns
        ------

        batch_ : QLM Batch
            QLM Batch with the jobs

        """
        # Deep Copy of the input batch object
        batch_ = copy.deepcopy(batch)
        if "weights" not in batch_.meta_data:
            raise ValueError(
                "The meta_data of the Batch do not have weights key")
        if "features" not in batch_.meta_data:
            raise ValueError(
                "The meta_data of the Batch do not have feature key")
        # Select the weight parameter names from batch_
        weights_name = batch_.meta_data["weights"]
        features_name = batch_.meta_data["features"]
        parameters = dict(zip(weights_name, self.weights))
        parameters.update(dict(zip(features_name, self.features)))

        # Iterating over the input jobs
        new_jobs = []
        for job in batch_:
            job.circuit = job.circuit(**parameters)
        #print("Finalizo Batch")
        return batch_
    def post_process(self, batch_result):
        #print("Post Procces")
        return batch_result

class ViewPlugin(AbstractPlugin):
    """
    Pluging for depicting the circuits of an input batch

    Parameters
    ----------

    plugin : str
        Name of the Before Plugin
    """
    def __init__(self, plugin=None):
        """
        Init method
        """
        self.plugin = plugin
    def compile(self, batch, hardware_specs):
        """
        compile method
        """
        print("Quantum Circuits from Pluging: {} \n".format(self.plugin))
        for j in batch:
            print(j.meta_data)
            print(j.circuit.display())
        return batch


def compute_pdf_from_pqc(batch, parameters):
    """
    Given a QLM Batch with a PQC representing a Multivariate
    Cumulative Distribution Function (cdf) creates all the mandatory
    PQCs for computing the corresponding Probability Distribution
    Function, pdf. The returned is a QLM Batch with the jobs mandatory
    for computing the pdf

    Parameters
    ----------

    batch : QLM Batch
        QLM batch with the Jobs to execute
    parameters : list
        list with the name of the features for pdf computation

    Returns
    ------

    batch_ : QLM Batch
        QLM Batch with the jobs for pdf copmputation
    """
    batch_ = copy.deepcopy(batch)
    if len(batch_) != 1:
        raise ValueError("BE AWARE: Input batch MUST HAVE only 1 job")


    # Select the first job
    job = [batch_[0]]
    for feature in parameters:
        temp_list = []
        for step_job in job:
            temp_list = temp_list + step_job.differentiate(feature)
        # Overwrite the input job list with the new job list
        job = temp_list

    # Overwrite the jobs of the output Batch
    batch_.jobs = temp_list
    return batch_

class pdfPluging(AbstractPlugin):
    """
    Given a batch with a PQC for computing a CDF this plugin computes
    the Batch for computing the corresponding PDF

    Parameters
    ----------

    parameters : list
        Name of all the features of the PQC. Mandatory for computing
        the Batch that allows the PDF
    """
    def __init__(self, parameters):
        """
        Init method
        """
        self.parameters = parameters
    def compile(self, batch, hardware_specs):
        """
        compile method
        """
        pdf_batch = compute_pdf_from_pqc(batch, self.parameters)
        return pdf_batch
    def post_process(self, batch_result):
        #print("Post Procces")
        output = np.array([r_.value for r_ in batch_result])
        final_result = Result()
        final_result.value = np.sum(output)
        final_result.meta_data = {
            "output" : output,
        }
        return BatchResult(results=[final_result])



class SetWeightsPlugin(AbstractPlugin):
    """
    Pluging for setting the weights of the different PQCs.
    The Batch MUST have a meta_data with at least the keys:
    fetures and weights

    Parameters
    ----------

    weights : numpy array
        Array with the weights of the model
    """

    def __init__(self, weights):
        """
        Init method
        """
        self.weights = weights
        #print(self.weights)

    def compile(self, batch, hardware_specs):
        """
        Loop over all the jobs of the input Batch and overwrite
        the different weights of the PQC

        Parameters
        ----------

        batch : QLM Batch
            QLM Batch object with the jobs to execute

        Returns
        ------

        batch_ : QLM Batch
            QLM Batch with the jobs

        """
        # Deep Copy of the input batch object
        batch_ = copy.deepcopy(batch)
        if "weights" not in batch_.meta_data:
            raise ValueError(
                "The meta_data of the Batch do not have weights key")
        if "features" not in batch_.meta_data:
            raise ValueError(
                "The meta_data of the Batch do not have feature key")
        # Select the weight parameter names from batch_
        weights_name = batch_.meta_data["weights"]
        weights_ = {k:v for k, v in zip(weights_name, self.weights)}
        # Iterating over the input jobs
        new_jobs = []
        for job in batch_:
            job.circuit = job.circuit(**weights_)
        return batch_

class GradientPlugin(AbstractPlugin):
    """
    Pluging for creating gradient Jobs.

    The input Batch for the compile method SHOULD HAVE only 1 job.
    The circuit of the job should be a parametric QLM circuit.
    The Batch MUST have a meta_data with at least the keys:
    fetures and weights


    """
    def __init__(self, variables, gradient_computation):
        """
        Init method
        """
        # Fixed the variables for derivative
        self.variables = variables
        # If is for computing gradient or not
        # If variables are features gradient_computation: False
        # If variables are weights gradient_computation: True
        self.gradient_computation = gradient_computation
        #print(self.weights)
    def compile(self, batch, hardware_specs):
        """
        Compile method of the plugin.

        Parameters
        ----------

        batch : QLM Batch
            QLM batch with the Jobs to execute

        Returns
        ------

        batch_ : QLM Batch
            QLM Batch with the jobs for computing graidents
        """

        #Deep Coopy of the input batch object
        batch_ = copy.deepcopy(batch)
        if len(batch_) != 1:
            raise ValueError("BE AWARE: Input batch MUST HAVE only 1 job")

        if "weights" not in batch_.meta_data:
            raise ValueError(
                "The meta_data of the Batch do not have weights key")
        if "features" not in batch_.meta_data:
            raise ValueError(
                "The meta_data of the Batch do not have feature key")

        # Select the first job
        job = batch_[0]
        weights_name = batch_.meta_data["weights"]
        list_of_jobs = []
        for parameter in self.variables:
            # For each parameter get the corresponding diferentiate jobs
            job_ = job.differentiate(parameter)
            for i_, step_job in enumerate(job_):
                step_job.meta_data = {
                    "parameter" : parameter,
                    "number" : i_,
                    "gradient_circuit" : self.gradient_computation

                }
                list_of_jobs.append(step_job)
        # list_of_jobs.append(job)
        batch_.jobs = list_of_jobs
        return batch_



class ProccesResultPluging(AbstractPlugin):
    """
    Procces the results of a input sample batch. The batch correspond
    to all the circuits mandatory for proccessing on sample
    of the dataset.
    """
    def compile(self, batch, hardware_specs):
        """
        The Batch pass throug the Pluging without modification
        """
        return batch

    def post_process(self, batch_result):
        """
        Given a QLM BatchResult object procces it
        Parameters
        ----------

        batch_result : QLM BatchResult
            QLM BatchResult object with the results to procces
        """
        prediction = []
        gradients_computation = False

        weights = {key:0.0 for key in  batch_result.meta_data["weights"]}
        weights2 = {key:[] for key in  batch_result.meta_data["weights"]}
        complete_weights = []
        for result_ in batch_result:
            if result_.meta_data["parameter"] == "prediction":
                prediction.append(result_.value)
            else:
                gradients_computation = True
                parameter_name = result_.meta_data["parameter"]
                weight_ = weights[parameter_name] + result_.value
                weights2.update(
                    {parameter_name:weights2[parameter_name] + [result_.value]}
                )
                weights.update({parameter_name:weight_})
        #print(weights, prediction)
        final_result = Result()
        final_result.meta_data = {
            "prediction" : prediction,
            "gradients" : weights,
            "gradients_computation": gradients_computation,
            "complete_weights" : weights2
        }
        return BatchResult(results=[final_result])


class MyQPU(QPUHandler):
    """
    New QPU. Update the result with the corresponding
    metadata of the input job

    Parameters
    ----------
    input_qpu : QLM qpu
        A qpu that will be the base for the new QPU
    """

    def __init__(self, input_qpu):
        """
        Init method
        """
        super().__init__()
        self.qpu = input_qpu


    def submit_job(self, job):
        """
        Given a job submit to the input_qpu and update
        the meta_data of the QLM Result object with the
        meta_data of the job
        """
        #job.circuit.display()
        result = self.qpu.submit(job)
        if not isinstance(result, Result):
            result = result.join()

        if result.meta_data is None:
            result.meta_data = {}

        if job.meta_data["gradient_circuit"] == False:
            result.meta_data.update({"parameter": "prediction"})

        if job.meta_data["gradient_circuit"] == True:
            result.meta_data.update({
                "parameter": job.meta_data["parameter"]
            })

        return result

class ProccesOnInputJunction(Junction):
    def __init__(self, feature_sample):
        """
        Init method
        """
        super().__init__(collective=True)
        self.feature_sample = feature_sample
    def run(self, batch, meta_data=None):
        """
        Loop over all the jobs of the input Batch and overwrite the
        different weights of the PQC
        """

        batch_ = copy.deepcopy(batch)
        # print("In ProccesOnInputJunction")
        # print(len(batch_))
        # print(batch_.meta_data)
        # print("In ProccesOnInputJunction")
        features_name = batch_.meta_data["features"]
        #features_sample = batch_.meta_data["feature_sample"]

        for job in batch_.jobs:
            feature_dict = dict(zip(features_name, self.feature_sample))
            job.circuit = job.circuit(**feature_dict)
        result = self.execute(batch_)
        gradients = None
        if result[0].meta_data["gradients_computation"]:
            gradients = result[0].meta_data["gradients"]
        result_ = Result()
        result_.meta_data = {
            "prediction" : sum(result[0].meta_data["prediction"]),
            "prediction_intermediate" : result[0].meta_data["prediction"],
            "gradient_dict":gradients

        }
        result_.meta_data.update(
            {"complete_weights" : result[0].meta_data["complete_weights"]}
        )

        return BatchResult(results=[result_])





class new_DataLoadingJunction(Junction):
    def __init__(self, features):
        """
        Init method
        """
        super().__init__(collective=True)
        self.features = features
    def run(self, batch, meta_data=None):
        """
        Loop over all the jobs of the input Batch and overwrite the
        different weights of the PQC
        """
        if "weights" not in batch.meta_data:
            raise ValueError(
                "The meta_data of the Batch do not have weights key")
        if "features" not in batch.meta_data:
            raise ValueError(
                "The meta_data of the Batch do not have feature key")

        # Select the string for looking the feature parameters
        name = batch.meta_data["features"]
        weights_name = batch.meta_data["weights"]

        # Iterating over the features
        predictions = []
        gradients = []
        for x_ in self.features:
            # Set the feature parameters of all the circuits with the
            # current sample features
            #new_jobs = []
            batch_ = copy.deepcopy(batch)
            batch_.meta_data.update({"feature_sample": x_})
            # Send the batch to the rest of the stact
            # print("In new_DataLoadingJunction")
            # print(batch_.meta_data)
            # print("In new_DataLoadingJunction")
            result = self.execute(batch_)
# 
#             # Procces the result
#             if len(result) == 0:
#                 raise ValueError("Empty results of the executed")
#             if len(result) > 1:
#                 raise ValueError(
#                     "The results of the executed batch is higher than 1")
#             predictions.append(result[0].meta_data["prediction"])
#             if result[0].meta_data["gradients_computation"]:
#                 gradients.append(result[0].meta_data["gradients"])
# 
#         result_ = Result()
#         # print(batch.meta_data)
#         if len(gradients) != 0:
#             gradients_ = np.array([
#                 [step_grad[wn] for wn in weights_name]
#                 for step_grad in gradients
#             ])
#         else:
#             gradients_ = None
#         result_.meta_data = {
#             "predictions" : np.vstack(predictions),
#             "gradients" : gradients_,
#             "gradient_dict":gradients
# 
#         }
# 
#         return BatchResult(results=[result_])

class DataLoadingJunction(Junction):
    """
    Pluging for loading Data into the circuits of the batch

    Parameters
    ----------

    X : numpy array
        Numpy array with the Features
    """
    def __init__(self, features):
        """
        Init method
        """
        super().__init__(collective=True)
        self.features = features

    def run(self, batch, meta_data=None):
        """
        Loop over all the jobs of the input Batch and overwrite the
        different weights of the PQC
        """
        if "weights" not in batch.meta_data:
            raise ValueError(
                "The meta_data of the Batch do not have weights key")
        if "features" not in batch.meta_data:
            raise ValueError(
                "The meta_data of the Batch do not have feature key")

        # Select the string for looking the feature parameters
        name = batch.meta_data["features"]
        weights_name = batch.meta_data["weights"]
        # print(batch_.meta_data)

        # Iterating over the features
        predictions = []
        gradients = []
        for x_ in self.features:
            # Set the feature parameters of all the circuits with the
            # current sample features
            #new_jobs = []
            batch_ = copy.deepcopy(batch)
            for job in batch_.jobs:
                # deep copy of the job for keeping the meta_data info
                # Get the dictionary with the parameters and the values
                feature_dict = dict(zip(name, x_))
                # Overwrite the circuit feature parameters
                # with the corresponding feature
                job.circuit = job.circuit(**feature_dict)
            # Send the batch to the rest of the stact
            result = self.execute(batch_)

            # Procces the result
            if len(result) == 0:
                raise ValueError("Empty results of the executed")
            if len(result) > 1:
                raise ValueError(
                    "The results of the executed batch is higher than 1")
            predictions.append(result[0].meta_data["prediction"])
            if result[0].meta_data["gradients_computation"]:
                gradients.append(result[0].meta_data["gradients"])

        result_ = Result()
        # print(batch.meta_data)
        if len(gradients) != 0:
            gradients_ = np.array([
                [step_grad[wn] for wn in weights_name]
                for step_grad in gradients
            ])
        else:
            gradients_ = None
        result_.meta_data = {
            "predictions" : np.vstack(predictions),
            "gradients" : gradients_,
            "gradient_dict":gradients

        }

        return BatchResult(results=[result_])
