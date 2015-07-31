#include "utils.h"


/*
	create new event
*/
static int oxnn_event_createEvent(lua_State *L)
{
	cudaEvent_t *event = (cudaEvent_t *)lua_newuserdata(L, sizeof(cudaEvent_t));
	luaL_newmetatable(L, "event.mt");
	lua_setmetatable(L, -2);
	cudaError_t  err   = cudaEventCreate(event);
	
	if (err!=cudaSuccess){
		printf ("error in eventCreate: %s \n", cudaGetErrorString(err));
		THError("aborting");	
	}
	return 1;
}


/*
	destroy givent event
*/

static int oxnn_event_destroyEvent(lua_State *L)
{
	cudaEvent_t *event = (cudaEvent_t *) lua_touserdata(L, 1);
	cudaError_t err = cudaEventDestroy(*event);
	if (err!=cudaSuccess){
		printf ("error in eventDestroy: %s \n", cudaGetErrorString(err));
		THError("aborting");	
	}
	return 1;
}

/*
	Provided an event, system will record
	down the current stream onto the stated event.
*/
static int oxnn_event_recordEvent(lua_State *L)
{
	THCState *state = getCutorchState(L);
	cudaEvent_t *event = (cudaEvent_t *) lua_touserdata(L, 1);
	cudaStream_t stream = state->currentStream;
	cudaError_t  err = cudaEventRecord(*event, stream);
	if (err!=cudaSuccess){
		printf ("error in eventRecord: %s \n", cudaGetErrorString(err));
		THError("aborting");	
	}
	return 1;
}

/*
	Provided an event idx, system will push
	a synchronization op onto current stream
*/
static int oxnn_event_streamWaitEvent(lua_State *L)
{
	THCState *state = getCutorchState(L);
	cudaEvent_t *event = (cudaEvent_t *) lua_touserdata(L, 1);
	cudaStream_t stream = state->currentStream;
	cudaError_t err = cudaStreamWaitEvent(stream, *event, 0);
	if (err!=cudaSuccess){
		printf ("error in streamWait: %s \n", cudaGetErrorString(err));	
	}
	return 1;

}

static const struct luaL_Reg oxnn_event__[] = {
	{"oxnn_event_createEvent", oxnn_event_createEvent},
	{"oxnn_event_destroyEvent",oxnn_event_destroyEvent},
	{"oxnn_event_recordEvent", oxnn_event_recordEvent},
	{"oxnn_event_streamWaitEvent", oxnn_event_streamWaitEvent},
   	{NULL, NULL}

};
/*TODO: implement some clean up functions*/

static void oxnn_event_init(lua_State *L) {

  lua_getglobal(L, "oxnn");
  luaL_register(L, NULL, oxnn_event__);

  lua_pop(L, 1);
}

