  const { createApp } = Vue
 
  createApp({

    data() {
      return {
        fileName: null,
        file: null,
        video: null,
        url: null,
        scenes: null,
        loading: false,
        errors: false,
        hashFile: false,
        videoExt: 'mp4',
        delete_cache: false,
      }
    },

    methods: {
        reset(){
          this.fileName = null
          this.file = null 
          this.video = null
          this.scenes = null
          this.loading = false
          this.$refs.inputFile.value=null
          this.errors = false,
          this.hashFile = false,
          this.url = null,
          this.delete_cache = false
        },

        handleFile(e) {
          const files = Array.from(e.target.files) || null
          this.setFile(files)
        },

        dropFile(e){
          const files = e.dataTransfer.files
          this.setFile(files)
        },

        setFile(files){
          if(files && files.length > 0){
            this.fileName = files[0].name
            this.file = files[0]
          }
        },

        handleUpload(){
          this.video = null
          this.scenes = null
          this.url = null
          let data = new FormData()
          data.append('file', this.file)
          data.append('no_cache', this.delete_cache)
          this.makeRequest('/predict', data)
   
        },

        handleUrlYoutube(){
          this.video = null
          this.scenes = null
          let data = new FormData()
          data.append('url', this.url) 
          data.append('no_cache', this.delete_cache)
          this.makeRequest('/predict_youtube', data)
        
        },

        async makeRequest(url, data){

          this.loading = true
          try {
            const response = await fetch(url, {
              method: 'POST',
              body: data
            })

            const jsonData = await response.json()

            if (response.ok) {
              this.processData(jsonData)
            } else {
              this.errors = jsonData['error']
            }
            
          } catch(error) {
               this.errors = error
          }

          this.loading = false

        },

        processData(jsonData){
          this.hashFile = jsonData['file_name']
          this.video = '/static/uploads/' + jsonData['dir'] + '/' + this.hashFile
          this.videoExt = this.hashFile.split('.').pop();
          this.scenes = jsonData['scenes']
          this.errors = false
          this.url = null
          this.fileName = null
          this.$refs.inputFile.value=null
        },

        getSeconds(time){
          const split_time = time.split(':')
          seconds = parseFloat(split_time[2])
          min = parseInt(split_time[1])
          hours = parseInt(split_time[0])
          return seconds + min * 60 + hours * 60 * 60

        }

    },
    delimiters: ['[[',']]']
  }).mount('#app')
