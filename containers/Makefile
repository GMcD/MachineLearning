build:
	docker-compose -f nn-build.yml build

run: build
	docker-compose -f nn-app.yml up

stop:
	docker-compose -f nn-app.yml down
