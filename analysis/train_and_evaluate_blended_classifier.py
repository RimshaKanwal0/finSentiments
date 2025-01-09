import time
import platform
import os
import psutil
import multiprocessing
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score
from utils import bytes_to_mb_or_gb


def train_and_evaluate_blended_classifier(classifier_combo, additional_info, num_classifiers):
    print(f"Blending the following classifiers: {classifier_combo}")

    start_time = time.time()  # Record the start time
    start_time = time.time()
    # Get CPU information
    num_cores = multiprocessing.cpu_count()
    processor_type = platform.processor()
    # Get OS information
    os_name = platform.system()+'('+platform.release()+')'
    # Get RAM information
    ram_total = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    ram_total_mb = ram_total / (1024**2)
    # Calculate CPU and RAM resources before training
    cpu_before = psutil.cpu_percent()
    available_memory_before = psutil.virtual_memory().available

################################################################################
    # Create the voting classifier with the selected combination
    selected_classifiers = [(name, classifiers[name]) for name in classifier_combo]
    voting_classifier = VotingClassifier(estimators=selected_classifiers, voting='hard')

    # Train the voting classifier
    voting_classifier.fit(X_train_tfidf_dense, y_train)

    # Make final predictions on the test data using the voting classifier
    final_predictions = voting_classifier.predict(X_test_tfidf_dense)

    # Calculate evaluation metrics for the blended model
    accuracy = accuracy_score(y_test, final_predictions)
    precision = precision_score(y_test, final_predictions, average='weighted', zero_division=1)
    recall = recall_score(y_test, final_predictions, average='weighted')
    f1 = f1_score(y_test, final_predictions, average='weighted')
    cm = confusion_matrix(y_test, final_predictions)

    # Calculate Cohen's kappa
    kappa = cohen_kappa_score(y_test, final_predictions)

################################################################################

    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate the execution time
    # Calculate CPU and RAM resources after training
    cpu_after = psutil.cpu_percent()
    available_memory_after = psutil.virtual_memory().available

    # Calculate resource usage during this iteration
    cpu_usage = max(cpu_after - cpu_before, 0)
    memory_usage = max(available_memory_after - available_memory_before, 0)  # Ensure non-negative value
    memory_usage_str = bytes_to_mb_or_gb(memory_usage)

    print(f"CPU Usage: {cpu_usage:.2f}%")
    print(f"Memory Usage: {memory_usage_str}")

    # Store the results in the list as a dictionary
    results_dict = {
        'Blended Classifiers': classifier_combo,
        'Accuracy': accuracy,
        'Kappa': kappa,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Confusion Matrix': cm,
        'Execution Time (s)': execution_time,
        'Total Classifiers': additional_info['Total Classifiers'],
        'Total Features': additional_info['Total Features'],
        'Training Data Size': additional_info['Training Data Size'],
        'Test Data Size': additional_info['Test Data Size'],
        'Random State': additional_info['Random State'],
        'Preprocessing': additional_info['Preprocessing'],
        'SMOTE': additional_info['SMOTE'],
        'Total CPU Cores': num_cores,
        'CPU Usage (%)': cpu_usage,
        'Total RAM': ram_total_mb,
        'Memory Usage': memory_usage_str,
        'Processor Type': processor_type,
        'OS': os_name
    }
    print(results_dict)


    return results_dict
