#export QUERY_DEFAULTS_LIMIT=25
#export AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
##export PERSISTENCE_DATA_PATH="/my/weaviate/data"
##export PERSISTENCE_DATA_PATH="$HOME/weaviate_data"
#export ENABLE_API_BASED_MODULES=true
#
#unset CLUSTER_HOSTNAME

cd ~/weaviate
#~/weaviate/weaviate-server --host 0.0.0.0 --port 8080 --scheme HTTP
env -i ~/weaviate/weaviate-server --host 0.0.0.0 --port 8080 --scheme http
